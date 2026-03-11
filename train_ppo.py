"""
On-policy PPO training loop with batched Brax environments.

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


def collect_rollouts(env, agent, rng_key, n_envs, n_steps, gamma, gae_lambda):
    """
    Collect n_steps transitions from all n_envs with the current policy,
    compute GAE advantages, and return a flat rollout buffer dict.

    Returns:
        rollout: dict with keys obs, actions, log_probs, advantages, returns
                 each shaped (n_steps * n_envs, ...)
        rng_key: updated PRNG key
    """
    rng_key, reset_key = jax.random.split(rng_key)
    rngs = jax.random.split(reset_key, n_envs)
    state = env.reset(rngs)

    obs_buf     = np.zeros((n_steps, n_envs, env.observation_size), np.float32)
    action_buf  = np.zeros((n_steps, n_envs, env.action_size),      np.float32)
    logprob_buf = np.zeros((n_steps, n_envs),                        np.float32)
    reward_buf  = np.zeros((n_steps, n_envs),                        np.float32)
    done_buf    = np.zeros((n_steps, n_envs),                        np.float32)
    value_buf   = np.zeros((n_steps, n_envs),                        np.float32)

    for t in range(n_steps):
        obs = np.array(state.obs, dtype=np.float32)  # (n_envs, obs_dim)
        rng_key, subkey = jax.random.split(rng_key)
        actions, log_probs, values = agent.get_action_logprob_value(obs, rng_key=subkey)

        obs_buf[t]     = obs
        action_buf[t]  = actions
        logprob_buf[t] = log_probs
        value_buf[t]   = values

        state = env.step(state, jnp.asarray(actions))
        reward_buf[t] = np.array(state.reward)
        done_buf[t]   = np.array(state.done)

    # Bootstrap V(s_{T+1}) for the last state
    last_value = agent.get_value(np.array(state.obs, dtype=np.float32))  # (n_envs,)

    # Compute GAE in reverse
    advantages = np.zeros_like(value_buf)
    last_gae = np.zeros(n_envs, np.float32)
    for t in reversed(range(n_steps)):
        nxt_val   = last_value if t == n_steps - 1 else value_buf[t + 1]
        nxt_nterm = 1.0 - done_buf[t]
        delta     = reward_buf[t] + gamma * nxt_val * nxt_nterm - value_buf[t]
        last_gae  = delta + gamma * gae_lambda * nxt_nterm * last_gae
        advantages[t] = last_gae
    returns = advantages + value_buf

    def flat(x):
        s = x.shape
        return x.reshape(s[0] * s[1], *s[2:]) if len(s) > 2 else x.reshape(-1)

    return {
        'obs':        flat(obs_buf),
        'actions':    flat(action_buf),
        'log_probs':  flat(logprob_buf),
        'advantages': flat(advantages),
        'returns':    flat(returns),
    }, rng_key


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

        n_envs         = int(getattr(self.cfg, 'n_envs', 16))
        episode_length = int(getattr(self.cfg, 'brax_episode_length', 1000))
        self.n_envs    = n_envs
        self.n_steps   = int(getattr(self.cfg, 'ppo_n_steps', 2048))
        self.gae_lambda = float(getattr(self.cfg, 'ppo_gae_lambda', 0.95))

        self.env = make_brax_env_batched(self.cfg.task_name, n_envs, episode_length)

        obs_shape    = (self.env.observation_size,)
        action_shape = (self.env.action_size,)
        agent_cfg    = self.cfg.agent

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

        utils.save_cfg(self.cfg, self.work_dir)
        utils.save_git_sha(self.work_dir)

    def train(self):
        gamma      = float(self.cfg.discount)
        eval_every = utils.Every(int(self.cfg.eval_every_frames))
        save_every = utils.Every(int(self.cfg.save_every_frames))
        train_until = utils.Until(int(self.cfg.num_train_frames))

        while train_until(self._global_step):
            rollout, self.rng_key = collect_rollouts(
                self.env, self.agent, self.rng_key,
                self.n_envs, self.n_steps, gamma, self.gae_lambda,
            )
            frames  = self.n_steps * self.n_envs
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

        # Always save at the end
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


@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg):
    log = logging.getLogger(__name__)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        log.info(f"Running on GPU {device_id}.")
    except omegaconf.errors.MissingMandatoryValue:
        pass

    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pkl'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
