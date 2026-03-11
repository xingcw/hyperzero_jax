"""
TD3 training with vectorized Brax environments and in-memory replay buffer.

Optimizations over the original:
  • N parallel envs (n_envs, default 64) — one JAX call per meta-step
  • In-memory circular replay buffer — no disk I/O during training
  • pmap across all TPU/GPU cores for gradient computation
  • JIT-compiled actor forward pass for fast batch action selection
  • Real-time progress display (FPS, step, losses)
"""

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import platform
import logging
import pickle
import time

if platform.system() == 'Linux':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MUJOCO_GL'] = 'disable'

from pathlib import Path

import hydra
import omegaconf
import numpy as np
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import utils.utils as utils
from utils.logger import Logger
from utils.env_common import ArraySpec

# If using multirun, set GPUs here
AVAILABLE_GPUS = [0, 1, 2, 3, 4]

log = logging.getLogger(__name__)


def make_agent(obs_spec, action_spec, cfg, device=None):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    if device is not None:
        cfg.device = device
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        self.rng_key = utils.set_seed_everywhere(cfg.seed)
        self._global_step = 0
        self._global_episode = 0
        self.timer = utils.Timer()
        self.setup()

    def setup(self):
        utils.assert_agent(self.cfg['agent_name'], self.cfg['pixel_obs'])
        simulator = getattr(self.cfg, 'simulator', 'brax')
        if simulator != 'brax':
            raise ValueError('Only simulator=brax is supported.')
        if self.cfg.pixel_obs:
            raise ValueError('Brax is state-only; set pixel_obs=false.')

        self.logger = Logger(self.work_dir)
        self.plot_dir = self.work_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        n_devices = jax.device_count()
        n_envs = int(getattr(self.cfg, 'n_envs', 64))
        brax_episode_length = getattr(self.cfg, 'brax_episode_length', 1000)

        log.info(f"JAX devices: {n_devices}  |  n_envs: {n_envs}")

        # Vectorized training env (N parallel envs, auto-reset)
        from utils.brax import make_vectorized
        self.train_env = make_vectorized(
            self.cfg.task_name, n_envs, brax_episode_length, self.cfg.seed
        )
        self.n_envs = self.train_env.n_envs  # may differ if n_envs % n_devices != 0
        log.info(f"Effective n_envs: {self.n_envs} ({self.n_envs // n_devices} per device)")

        # (eval fn built after agent, below)

        # In-memory circular replay buffer
        from utils.replay_buffer import CircularReplayBuffer
        self.replay_buffer = CircularReplayBuffer(
            capacity=int(self.cfg.replay_buffer_size),
            obs_dim=self.train_env.obs_size,
            action_dim=self.train_env.action_size,
        )

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.agent,
        )

        # Scan-based batched eval: all episodes in parallel, lax.scan over steps
        from utils.brax import make_scan_eval
        self._jit_eval, self._eval_rng = make_scan_eval(
            self.cfg.task_name,
            int(self.cfg.num_eval_episodes),
            brax_episode_length,
            self.agent.actor_module.apply,
            self.cfg.seed,
        )

        utils.save_cfg(self.cfg, self.work_dir)
        utils.save_git_sha(self.work_dir)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self._global_step * self.cfg.action_repeat

    def eval(self):
        # All episodes run in parallel on-device; lax.scan replaces the step loop.
        mean_reward, self._eval_rng = self._jit_eval(
            self.agent.actor_state.params, self._eval_rng
        )
        self._global_episode += self.cfg.num_eval_episodes

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as lg:
            lg('episode_reward', float(mean_reward))
            lg('episode_length', self.cfg.brax_episode_length)
            lg('episode', self._global_episode)
            lg('step', self.global_step)

    def train(self, task_id=1):
        # Training predicates
        total_frames = self.cfg.num_train_frames * task_id
        seed_frames = (self.cfg.num_seed_frames +
                       self.cfg.num_train_frames * (task_id - 1))
        train_until = utils.Until(total_frames, self.cfg.action_repeat)
        seed_until = utils.Until(seed_frames, self.cfg.action_repeat)
        eval_every = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        save_every = utils.Every(self.cfg.save_every_frames, self.cfg.action_repeat)
        discount = float(self.cfg.discount)
        batch_size = int(self.cfg.batch_size)
        n_envs = self.n_envs
        action_repeat = self.cfg.action_repeat

        # Reset all envs
        obs = self.train_env.reset()  # (n_envs, obs_dim)
        metrics = {}
        fps_timer = time.time()
        fps_step = 0

        while train_until(self.global_step):
            # ---- Action selection ----
            if seed_until(self.global_step):
                actions = np.random.uniform(
                    -1.0, 1.0,
                    size=(n_envs, self.train_env.action_size),
                ).astype(np.float32)
            else:
                actions = self.agent.act_batch(obs, self.global_step, eval_mode=False)

            # ---- Step all envs (single JAX call, O(1) time) ----
            next_obs, rewards, dones = self.train_env.step(actions)

            # ---- Add batch to in-memory replay buffer ----
            self.replay_buffer.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs

            # ---- Update agent ----
            if (not seed_until(self.global_step) and
                    len(self.replay_buffer) >= batch_size):
                # One pmap call: lax.scan does all n_envs updates on-device
                big_batch = self.replay_buffer.sample(batch_size * n_envs)
                metrics = self.agent.update_many(
                    big_batch, n_updates=n_envs,
                    step=self.global_step, discount=discount,
                )
                metrics['batch_reward'] = float(np.mean(rewards))
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            self._global_step += n_envs

            # ---- Compute FPS and log progress ----
            fps_step += n_envs
            now = time.time()
            if now - fps_timer >= 10.0:  # print every 10 s
                fps = fps_step / (now - fps_timer)
                fps_timer = now
                fps_step = 0
                reward_mean = float(np.mean(rewards))
                buf_size = len(self.replay_buffer)
                if seed_until(self.global_step):
                    print(
                        f"step {self.global_step:>10,} | "
                        f"FPS {fps:>6.0f} | "
                        f"buf {buf_size:>8,} | "
                        f"rew {reward_mean:>6.3f} | "
                        f"[seed phase — no updates yet]"
                    )
                else:
                    critic_loss = metrics.get('critic_loss', float('nan'))
                    actor_loss = metrics.get('actor_loss', float('nan'))
                    print(
                        f"step {self.global_step:>10,} | "
                        f"FPS {fps:>6.0f} | "
                        f"buf {buf_size:>8,} | "
                        f"rew {reward_mean:>6.3f} | "
                        f"critic {critic_loss:>7.4f} | "
                        f"actor {actor_loss:>7.4f}"
                    )

            # ---- Periodic eval / save ----
            if eval_every(self.global_step):
                self.logger.log(
                    'eval_total_time', self.timer.total_time(), self.global_frame
                )
                self.eval()

            if save_every(self.global_step):
                self.agent.save(self.model_dir, self.global_frame)
                if self.cfg.save_snapshot:
                    self.save_snapshot()

        # Final save
        self.agent.save(self.model_dir, self.global_frame)

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pkl'
        with snapshot.open('wb') as f:
            pickle.dump({
                '_global_step': self._global_step,
                '_global_episode': self._global_episode,
                'timer': self.timer,
                'agent_actor_params': jax.device_get(self.agent.actor_state.params),
                'agent_critic_params': jax.device_get(self.agent.critic_state.params),
                'agent_actor_target': jax.device_get(self.agent.actor_target_params),
                'agent_critic_target': jax.device_get(self.agent.critic_target_params),
            }, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pkl'
        with snapshot.open('rb') as f:
            payload = pickle.load(f)
        self._global_step = payload['_global_step']
        self._global_episode = payload['_global_episode']
        self.timer = payload['timer']
        self.agent.actor_state = self.agent.actor_state.replace(
            params=jax.device_put(payload['agent_actor_params'])
        )
        self.agent.critic_state = self.agent.critic_state.replace(
            params=jax.device_put(payload['agent_critic_params'])
        )
        self.agent.actor_target_params = jax.device_put(payload['agent_actor_target'])
        self.agent.critic_target_params = jax.device_put(payload['agent_critic_target'])
        self.agent._invalidate_replicated_states()


@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        logger.info(f"Running on GPU {device_id}.")
    except omegaconf.errors.MissingMandatoryValue:
        pass

    logger.info(f"JAX devices: {jax.devices()} ({jax.device_count()} total)")

    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pkl'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
