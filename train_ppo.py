"""
On-policy PPO training loop with batched Brax environments.

Multi-device training via jax.pmap; fast rollouts via jax.lax.scan.

Usage (same Hydra config system as train.py):
    python train_ppo.py agent@_global_=ppo task@_global_=cheetah_run \\
        reward@_global_=cheetah_default dynamics@_global_=default

After training, generate the rollout dataset with eval.py:
    python eval.py --workdir <results_dir> --eval_mode sl_data \\
        --rollout_dir <rollout_dir> --n_episodes 10

The saved model format (actor.pkl / critic.pkl) is the same as TD3,
so eval.py loads PPO agents with no changes.
"""

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
import os
import pickle
import platform
from functools import partial
from pathlib import Path

if platform.system() == 'Linux':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MUJOCO_GL'] = 'disable'

import numpy as np
import jax
import jax.numpy as jnp
import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import utils.utils as utils
from utils.logger import Logger
from agents.ppo import PPOAgent

AVAILABLE_GPUS = [0, 1, 2, 3, 4]

_TASK_NAME_MAP = {
    'cheetah_run': 'halfcheetah',
    'walker_walk': 'walker2d',
    'hopper_hop': 'hopper',
    'ant_run': 'ant',
}


def _brax_env_name(task_name):
    name = task_name.lower().replace('-', '_')
    return _TASK_NAME_MAP.get(name, name)


def make_brax_env_batched(task_name, n_envs, episode_length):
    """Create a batched Brax env (auto-reset) for on-policy rollout collection."""
    from brax import envs
    env = envs.create(
        env_name=_brax_env_name(task_name),
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=n_envs,
    )
    return env


def make_rollout_fn(env, agent, n_steps):
    """
    Build a pmap'd rollout function using jax.lax.scan.

    Each device independently collects n_steps transitions from n_envs_per_device
    environments. The scan replaces the Python step loop, keeping computation
    entirely on-device with no host-device transfers during rollout.

    Args:
        env: Brax env created with batch_size = n_envs_per_device.
        agent: PPOAgent instance (actor/critic modules captured in closure).
        n_steps: Number of steps to collect per device.

    Returns:
        pmap'd function: (actor_params_rep, critic_params_rep, device_rngs)
            → (traj, last_values)
          traj = (obs, actions, log_probs, values, rewards, dones)
          each of shape (n_devices, n_steps, n_envs_per_device, ...)
    """
    actor_apply = agent.actor_module.apply
    critic_apply = agent.critic_module.apply
    n_envs_per_device = env.batch_size

    @partial(jax.pmap, axis_name='devices')
    def _collect(actor_params, critic_params, device_rng):
        # Each device resets its own independent environments
        env_rngs = jax.random.split(device_rng, n_envs_per_device)
        init_state = env.reset(env_rngs)

        def step_fn(carry, _):
            state, rng = carry
            rng, subkey = jax.random.split(rng)
            obs = state.obs

            # Actor forward pass
            mean, log_std = actor_apply({'params': actor_params}, obs)
            std = jnp.exp(log_std)
            action = jnp.clip(
                mean + std * jax.random.normal(subkey, mean.shape), -1.0, 1.0
            )
            log_prob = (
                -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
            ).sum(-1)

            # Critic forward pass
            value = critic_apply({'params': critic_params}, obs)[..., 0]

            new_state = env.step(state, action)
            return (new_state, rng), (
                obs, action, log_prob, value, new_state.reward, new_state.done
            )

        (final_state, _), traj = jax.lax.scan(
            step_fn, (init_state, device_rng), None, length=n_steps
        )
        # Bootstrap value for the last observation
        last_value = critic_apply({'params': critic_params}, final_state.obs)[..., 0]
        return traj, last_value

    return _collect


def make_gae_fn(gamma, gae_lambda):
    """
    Build a JIT-compiled GAE computation function.

    Uses jax.lax.scan with reverse=True to replace the Python reverse loop,
    keeping the entire computation on-device.
    """
    @jax.jit
    def compute_gae(value_buf, reward_buf, done_buf, last_value):
        """
        Args:
            value_buf:  (n_steps, n_envs) V(s_t) estimates
            reward_buf: (n_steps, n_envs) rewards
            done_buf:   (n_steps, n_envs) episode-end flags
            last_value: (n_envs,) V(s_{T+1}) bootstrap
        Returns:
            advantages: (n_steps, n_envs)
            returns:    (n_steps, n_envs)
        """
        # next_values[t] = value_buf[t+1] for t < T-1, last_value for t = T-1
        next_values = jnp.concatenate([value_buf[1:], last_value[None]], axis=0)

        def scan_fn(last_gae, x):
            value, next_value, reward, done = x
            nxt_nterm = 1.0 - done
            delta = reward + gamma * next_value * nxt_nterm - value
            last_gae = delta + gamma * gae_lambda * nxt_nterm * last_gae
            return last_gae, last_gae

        n_envs = value_buf.shape[1]
        _, advantages = jax.lax.scan(
            scan_fn,
            jnp.zeros(n_envs),
            (value_buf, next_values, reward_buf, done_buf),
            reverse=True,
        )
        return advantages, advantages + value_buf

    return compute_gae


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        self.rng_key = utils.set_seed_everywhere(cfg.seed)
        self._global_step = 0
        self.timer = utils.Timer()
        self.setup()

    def setup(self):
        self.logger = Logger(self.work_dir)
        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        self.n_devices = jax.device_count()
        n_envs = int(getattr(self.cfg, 'n_envs', 16))
        episode_length = int(getattr(self.cfg, 'brax_episode_length', 1000))
        self.n_steps = int(getattr(self.cfg, 'ppo_n_steps', 2048))
        self.gae_lambda = float(getattr(self.cfg, 'ppo_gae_lambda', 0.95))

        # Ensure n_envs is divisible by n_devices
        self.n_envs_per_device = max(1, n_envs // self.n_devices)
        self.n_envs = self.n_envs_per_device * self.n_devices

        logging.getLogger(__name__).info(
            f"Devices: {self.n_devices}, envs/device: {self.n_envs_per_device}, "
            f"total envs: {self.n_envs}"
        )

        # Create env with per-device batch size; pmap handles device distribution
        self.env = make_brax_env_batched(
            self.cfg.task_name, self.n_envs_per_device, episode_length
        )

        obs_shape = (self.env.observation_size,)
        action_shape = (self.env.action_size,)
        agent_cfg = self.cfg.agent

        self.agent = PPOAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.cfg.device,
            lr=float(self.cfg.lr),
            hidden_dim=int(agent_cfg.hidden_dim),
            clip_eps=float(getattr(agent_cfg, 'clip_eps', 0.2)),
            value_coef=float(getattr(agent_cfg, 'value_coef', 0.5)),
            entropy_coef=float(getattr(agent_cfg, 'entropy_coef', 0.01)),
            n_epochs=int(getattr(agent_cfg, 'n_epochs', 10)),
            minibatch_size=int(getattr(agent_cfg, 'minibatch_size', 64)),
            max_grad_norm=float(getattr(agent_cfg, 'max_grad_norm', 0.5)),
        )

        # Build JIT+pmap rollout function and JIT GAE function
        self._rollout_fn = make_rollout_fn(self.env, self.agent, self.n_steps)
        self._compute_gae = make_gae_fn(
            gamma=float(self.cfg.discount),
            gae_lambda=self.gae_lambda,
        )

        utils.save_cfg(self.cfg, self.work_dir)
        utils.save_git_sha(self.work_dir)

    def collect_rollouts(self):
        """
        Collect rollouts across all devices via pmap + lax.scan,
        then compute GAE advantages with a JIT-compiled reverse scan.

        Returns: flat rollout dict with keys obs, actions, log_probs,
                 advantages, returns — each shaped (n_steps * n_envs, ...).
        """
        # Generate one PRNG key per device (devices get independent trajectories)
        self.rng_key, *device_rngs = jax.random.split(self.rng_key, self.n_devices + 1)
        device_rngs = jnp.stack(device_rngs)  # (n_devices, 2)

        # Replicate params to all devices
        actor_params_rep = jax.device_put_replicated(
            self.agent.actor_state.params, jax.devices()
        )
        critic_params_rep = jax.device_put_replicated(
            self.agent.critic_state.params, jax.devices()
        )

        # Run pmap'd lax.scan rollout
        # traj: each tensor is (n_devices, n_steps, n_envs_per_device, ...)
        # last_value: (n_devices, n_envs_per_device)
        traj, last_value = self._rollout_fn(
            actor_params_rep, critic_params_rep, device_rngs
        )
        jax.block_until_ready(traj)  # ensure timings are accurate

        obs_buf, action_buf, logprob_buf, value_buf, reward_buf, done_buf = traj

        # Merge device and env dims:
        # (n_devices, n_steps, n_envs_per_device, ...) → (n_steps, n_envs, ...)
        def merge(x):
            # swap to (n_steps, n_devices, n_envs_per_device, ...) then flatten
            x = x.swapaxes(0, 1)
            return x.reshape(self.n_steps, self.n_envs, *x.shape[3:])

        obs_buf     = merge(obs_buf)      # (n_steps, n_envs, obs_dim)
        action_buf  = merge(action_buf)   # (n_steps, n_envs, action_dim)
        logprob_buf = merge(logprob_buf)  # (n_steps, n_envs)
        value_buf   = merge(value_buf)    # (n_steps, n_envs)
        reward_buf  = merge(reward_buf)   # (n_steps, n_envs)
        done_buf    = merge(done_buf)     # (n_steps, n_envs)
        last_value  = last_value.reshape(self.n_envs)  # (n_envs,)

        # JIT-compiled GAE with reverse lax.scan
        advantages, returns = self._compute_gae(
            value_buf, reward_buf, done_buf, last_value
        )

        def flat(x):
            s = x.shape
            return x.reshape(s[0] * s[1], *s[2:]) if len(s) > 2 else x.reshape(-1)

        return {
            'obs':        flat(obs_buf),
            'actions':    flat(action_buf),
            'log_probs':  flat(logprob_buf),
            'advantages': flat(advantages),
            'returns':    flat(returns),
        }

    def train(self):
        gamma = float(self.cfg.discount)
        eval_every = utils.Every(int(self.cfg.eval_every_frames))
        save_every = utils.Every(int(self.cfg.save_every_frames))
        train_until = utils.Until(int(self.cfg.num_train_frames))

        while train_until(self._global_step):
            rollout = self.collect_rollouts()
            frames = self.n_steps * self.n_envs
            metrics = self.agent.update(rollout, self._global_step)
            elapsed, total = self.timer.reset()

            with self.logger.log_and_dump_ctx(self._global_step, ty='train') as log:
                log('episode_reward', float(rollout['returns'].mean()))
                log('fps', frames / elapsed)
                log('total_time', total)
                log('episode', self._global_step // frames)
                log('step', self._global_step)
                for k, v in metrics.items():
                    log(k, v)

            self._global_step += frames

            if save_every(self._global_step):
                self.agent.save(self.model_dir, self._global_step)

            if eval_every(self._global_step):
                self._eval()

        self.agent.save(self.model_dir, self._global_step)

    def _eval(self):
        """Quick evaluation rollout in single-env mode (compatible with BraxWrapper)."""
        from utils import brax as brax_utils
        try:
            reward_parameters = OmegaConf.to_container(self.cfg.reward_parameters)
        except omegaconf.errors.ConfigAttributeError:
            reward_parameters = {}
        try:
            dynamics_parameters = OmegaConf.to_container(self.cfg.dynamics_parameters)
        except omegaconf.errors.ConfigAttributeError:
            dynamics_parameters = {}

        eval_env = brax_utils.make(
            self.cfg.task_name, 1, 1, reward_parameters, dynamics_parameters,
            self.cfg.seed + 999, pixel_obs=False,
            episode_length=int(getattr(self.cfg, 'brax_episode_length', 1000)),
        )
        n_eps = int(getattr(self.cfg, 'num_eval_episodes', 5))
        total_reward = 0.0
        for _ in range(n_eps):
            ts = eval_env.reset()
            while not ts.last():
                action = self.agent.act(ts.observation, self._global_step, eval_mode=True)
                ts = eval_env.step(action)
                total_reward += ts.reward

        with self.logger.log_and_dump_ctx(self._global_step, ty='eval') as log:
            log('episode_reward', total_reward / n_eps)
            log('episode', self._global_step)
            log('step', self._global_step)

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pkl'
        with snapshot.open('wb') as f:
            pickle.dump({
                '_global_step': self._global_step,
                'actor_params': jax.device_get(self.agent.actor_state.params),
                'critic_params': jax.device_get(self.agent.critic_state.params),
                'timer': self.timer,
            }, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pkl'
        with snapshot.open('rb') as f:
            payload = pickle.load(f)
        self._global_step = payload['_global_step']
        self.timer = payload['timer']
        self.agent.actor_state = self.agent.actor_state.replace(
            params=jax.device_put(payload['actor_params'])
        )
        self.agent.critic_state = self.agent.critic_state.replace(
            params=jax.device_put(payload['critic_params'])
        )
        # Force re-replication of params on next update
        self.agent._invalidate_replicated_states()


@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg):
    log = logging.getLogger(__name__)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        log.info(f"Running on GPU {device_id}.")
    except omegaconf.errors.MissingMandatoryValue:
        pass

    log.info(f"JAX devices: {jax.devices()} ({jax.device_count()} total)")

    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pkl'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
