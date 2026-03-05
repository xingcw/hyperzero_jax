"""
Standard RL training loop.
Works on DMC with states and pixel observations.
"""

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import platform
import logging
import pickle

if platform.system() == 'Linux':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    # Disable MuJoCo rendering (not needed for physics-only on TPU)
    os.environ['MUJOCO_GL'] = 'disable'

from pathlib import Path

import hydra
import omegaconf
import numpy as np
import jax
from utils.env_common import ArraySpec
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import utils.utils as utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBufferStorage, make_replay_loader
# from utils.video import TrainVideoRecorder, VideoRecorder

# If using multirun, set the GPUs here:
AVAILABLE_GPUS = [0, 1, 2, 3, 4]


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
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # some assertions
        utils.assert_agent(self.cfg['agent_name'], self.cfg['pixel_obs'])
        simulator = getattr(self.cfg, 'simulator', 'brax')
        if simulator != 'brax':
            raise ValueError('Only simulator=brax is supported (DMC/suite disabled for TPU/headless).')
        if self.cfg.pixel_obs:
            raise ValueError('Brax is state-only; set obs@_global_: states (pixel_obs=false).')

        # create logger
        self.logger = Logger(self.work_dir)

        # get the reward parameters (used only for DMC)
        reward_parameters = OmegaConf.to_container(self.cfg.reward_parameters)

        # get the dynamics parameters (used only for DMC)
        dynamics_parameters = OmegaConf.to_container(self.cfg.dynamics_parameters)

        # create envs (Brax only; no DMC/suite for TPU/headless)
        from utils import brax as brax_env
        brax_episode_length = getattr(self.cfg, 'brax_episode_length', 1000)
        self.train_env = brax_env.make(
            self.cfg.task_name, self.cfg.frame_stack,
            self.cfg.action_repeat, reward_parameters,
            dynamics_parameters, self.cfg.seed, self.cfg.pixel_obs,
            episode_length=brax_episode_length,
        )
        self.eval_env = brax_env.make(
            self.cfg.task_name, self.cfg.frame_stack,
            self.cfg.action_repeat, reward_parameters,
            dynamics_parameters, self.cfg.seed, self.cfg.pixel_obs,
            episode_length=brax_episode_length,
        )
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      ArraySpec((1,), np.float32, 'reward'),
                      ArraySpec((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if self.cfg.save_video else None,
        #     fps=60 // self.cfg.action_repeat
        # )
        # self.train_video_recorder = TrainVideoRecorder(
        #     self.work_dir if self.cfg.save_train_video else None,
        #     fps=60 // self.cfg.action_repeat
        # )

        self.plot_dir = self.work_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        # save cfg and git sha
        utils.save_cfg(self.cfg, self.work_dir)
        utils.save_git_sha(self.work_dir)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            # self.video_recorder.init(self.eval_env, enabled=episode == 0)
            while not time_step.last():
                with utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                # self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            # self.video_recorder.save(f'{self.global_frame}_{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self, task_id=1):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames * task_id,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames + self.cfg.num_train_frames * (task_id - 1),
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        plot_every_step = utils.Every(self.cfg.plot_every_frames,
                                      self.cfg.action_repeat)
        save_every_step = utils.Every(self.cfg.save_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        # self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                # self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            if save_every_step(self.global_step):
                self.agent.save(self.model_dir, self.global_frame)

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # sample action
            with utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            # self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pkl'
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        # Save agent state separately
        payload['agent_actor_params'] = jax.device_get(self.agent.actor_state.params)
        payload['agent_critic_params'] = jax.device_get(self.agent.critic_state.params)
        payload['agent_actor_target'] = jax.device_get(self.agent.actor_target_params)
        payload['agent_critic_target'] = jax.device_get(self.agent.critic_target_params)
        with snapshot.open('wb') as f:
            pickle.dump(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pkl'
        with snapshot.open('rb') as f:
            payload = pickle.load(f)
        self.timer = payload['timer']
        self._global_step = payload['_global_step']
        self._global_episode = payload['_global_episode']
        self.agent.actor_state = self.agent.actor_state.replace(
            params=jax.device_put(payload['agent_actor_params'])
        )
        self.agent.critic_state = self.agent.critic_state.replace(
            params=jax.device_put(payload['agent_critic_params'])
        )
        self.agent.actor_target_params = jax.device_put(payload['agent_actor_target'])
        self.agent.critic_target_params = jax.device_put(payload['agent_critic_target'])


@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg):
    log = logging.getLogger(__name__)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        log.info(f"Total number of GPUs is {AVAILABLE_GPUS}, running on GPU {device_id}.")
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
