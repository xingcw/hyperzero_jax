import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import platform

if platform.system() == 'Linux':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    # Disable MuJoCo rendering (not needed for physics-only on TPU)
    os.environ['MUJOCO_GL'] = 'disable'

import argparse
from pathlib import Path

import numpy as np
import omegaconf
from omegaconf import OmegaConf
from collections import defaultdict

import jax
import jax.numpy as jnp

import utils.utils as utils
import utils.plots as plots
from train import make_agent
from train_rl_regressor import make_approximator
from utils.video import VideoRecorder


class Workspace:
    def __init__(self, cfg, work_dir, args):
        self.work_dir = Path(work_dir)
        self.base_name = self.work_dir.parents[0].name

        self.cfg = cfg
        self.args = args
        self.rng_key = utils.set_seed_everywhere(cfg.seed)

        # Video dir
        self.video_dir = Path(args.video_dir).joinpath(f'{self.work_dir.parents[1].name}')
        self.video_dir.mkdir(exist_ok=True, parents=True)

        self.setup()

        # create and load the RL agent
        self.agent = make_agent(self.eval_env_rl_agent.observation_spec(),
                                self.eval_env_rl_agent.action_spec(),
                                self.cfg.agent)
        self.step_to_load = args.step_to_load if args.step_to_load != 0 else utils.get_last_model(self.agent_model_dir)
        self.agent.load(self.agent_model_dir, self.step_to_load)

        if args.rl_regressor_workdir is not None and args.rl_regressor_workdir != 'None':
            # create and load the RL regressor
            rl_regressor_cfg_path = Path(args.rl_regressor_workdir).joinpath('cfg.yaml')
            rl_regressor_cfg = OmegaConf.load(rl_regressor_cfg_path)
            self.rl_regressor_name = rl_regressor_cfg.approximator_name
            self.input_to_regressor = rl_regressor_cfg.input_to_model
            self.rl_regressor_seed = rl_regressor_cfg.seed
            self.is_meta_learning = True if 'meta' in self.rl_regressor_name else False

            # Approximated RL rollout dir
            self.rollout_comparison_data = Path(f"{args.rollout_dir}_comparison").joinpath(self.input_to_regressor,
                                                                                           str(self.rl_regressor_seed),
                                                                                           self.cfg.task_name)
            self.rollout_comparison_data.mkdir(exist_ok=True, parents=True)

            # Overwrite video dir and video recorder
            self.video_dir = Path(args.video_dir).joinpath(self.input_to_regressor,
                                                           str(self.rl_regressor_seed), self.cfg.task_name)
            self.video_dir.mkdir(exist_ok=True, parents=True)
            self.video_recorder = VideoRecorder(
                self.video_dir,
                fps=60 // self.cfg.action_repeat
            )

            rl_regressor_work_dir = Path(args.rl_regressor_workdir)
            rl_regressor_model_dir = rl_regressor_work_dir / 'models'

            if self.input_to_regressor == 'rew':
                input_dim = self._get_reward_param_dim()
            elif self.input_to_regressor == 'dyn':
                input_dim = self._get_dynamics_param_dim()
            elif self.input_to_regressor == 'rew_dyn':
                input_dim = self._get_reward_dynamics_param_dim()
            else:
                raise NotImplementedError

            self.rl_regressor = make_approximator(input_dim,
                                                  self.eval_env_rl_agent.observation_spec().shape[0],
                                                  self.eval_env_rl_agent.action_spec().shape[0],
                                                  rl_regressor_cfg.approximator)
            regressor_step_to_load = utils.get_last_model(rl_regressor_model_dir)
            self.rl_regressor.load(rl_regressor_model_dir, regressor_step_to_load)
            if not hasattr(self.rl_regressor, 'act'):
                print("RL regressor does not have the policy.")
                self.rl_regressor = None
        else:
            print("Did not load the RL regressor.")

            # RL Rollout dir
            self.rollout_dir = Path(args.rollout_dir).joinpath(self.cfg.task_name)
            self.rollout_dir.mkdir(exist_ok=True, parents=True)

            self.rl_regressor = None
            self.rl_regressor_name = ''

    def setup(self):
        # get the reward parameters
        reward_parameters = OmegaConf.to_container(self.cfg.reward_parameters)

        # get the dynamics parameters
        try:
            dynamics_parameters = OmegaConf.to_container(self.cfg.dynamics_parameters)
        except omegaconf.errors.ConfigAttributeError:
            dynamics_parameters = {'use_default': True}

        # create envs (Brax only; no DMC/suite)
        from utils import brax as brax_env
        brax_episode_length = getattr(self.cfg, 'brax_episode_length', 1000)
        self.eval_env_rl_agent = brax_env.make(
            self.cfg.task_name, self.cfg.frame_stack,
            self.cfg.action_repeat, reward_parameters,
            dynamics_parameters, self.cfg.seed, self.cfg.pixel_obs,
            episode_length=brax_episode_length,
        )
        self.eval_env_rl_approx = brax_env.make(
            self.cfg.task_name, self.cfg.frame_stack,
            self.cfg.action_repeat, reward_parameters,
            dynamics_parameters, self.cfg.seed + 1, self.cfg.pixel_obs,
            episode_length=brax_episode_length,
        )
        self.domain = self.cfg.task_name.split('_')[0] if '_' in self.cfg.task_name else None

        self.video_recorder = VideoRecorder(
            self.video_dir,
            fps=60 // self.cfg.action_repeat
        )

        self.plot_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), 'eval_plots',
                                                          f'{self.work_dir.parents[1].name}')))
        self.plot_dir.mkdir(exist_ok=True, parents=True)
        self.agent_model_dir = self.work_dir / 'models'

    def rollout(self, n_episodes=1, use_approximator=False):
        rollout_data = defaultdict(list)
        env = self.eval_env_rl_approx if use_approximator else self.eval_env_rl_agent

        for episode in range(n_episodes):
            episode_rollout = defaultdict(list)
            time_step = env.reset()
            self.video_recorder.init(env, enabled=(episode == 0 and self.args.eval_mode == 'comparison_data'))

            while not time_step.last():
                with utils.eval_mode(self.agent):
                    reward_param = self._get_reward_param()
                    dynamics_param = self._get_dynamics_param()
                    reward_dynamics_param = self._get_reward_dynamics_param()

                    if use_approximator:
                        if self.input_to_regressor == 'rew':
                            action = self.rl_regressor.act(reward_param, time_step.observation)
                        elif self.input_to_regressor == 'dyn':
                            action = self.rl_regressor.act(dynamics_param, time_step.observation)
                        elif self.input_to_regressor == 'rew_dyn':
                            action = self.rl_regressor.act(reward_dynamics_param, time_step.observation)
                        else:
                            raise NotImplementedError
                    else:
                        action = self.agent.act(time_step.observation, self.step_to_load, eval_mode=True)
                        observed = self.agent.observe(time_step.observation, action)
                        episode_rollout['value'].append(observed['value'])

                    # save the trajectory
                    episode_rollout['reward_param'].append(reward_param)
                    episode_rollout['dynamics_param'].append(dynamics_param)
                    episode_rollout['state'].append(time_step.observation)
                    episode_rollout['discount'].append(time_step.discount)
                    episode_rollout['action'].append(action)
                    # Brax does not expose raw MuJoCo physics; store zero placeholders
                    episode_rollout['physics_qpos'].append(np.zeros(1, dtype=np.float32))
                    episode_rollout['physics_qvel'].append(np.zeros(1, dtype=np.float32))

                time_step = env.step(action)
                self.video_recorder.record(env)

                episode_rollout['reward'].append([time_step.reward])
                episode_rollout['next_state'].append(time_step.observation)

            if use_approximator:
                video_name = f'{self.base_name}_{self.rl_regressor_name}-approxseed-{self.rl_regressor_seed}.mp4'
            else:
                video_name = f'{self.base_name}.mp4'
            self.video_recorder.save(video_name)

            # concatenate across the current episode
            for k, v in episode_rollout.items():
                rollout_data[k].append(np.stack(v))

        # concatenate across all episodes
        for k, v in rollout_data.items():
            rollout_data[k] = np.stack(v)

        return rollout_data

    def finetune_meta_policy(self, data):
        n_episodes = 10
        state = data['state'][0:n_episodes, :, :].reshape(n_episodes * 1000, -1)
        action = data['action'][0:n_episodes, :, :].reshape(n_episodes * 1000, -1)
        batch_size = state.shape[0]

        if self.input_to_regressor == 'rew':
            input_param = self._get_reward_param()
        elif self.input_to_regressor == 'dyn':
            input_param = self._get_dynamics_param()
        elif self.input_to_regressor == 'rew_dyn':
            input_param = self._get_reward_dynamics_param()
        else:
            raise NotImplementedError
        input_param = np.repeat(input_param, batch_size).reshape(batch_size, -1)

        self.rl_regressor.finetune(input_param, state, action)

    def save_rollout(self, rollout_data, dir, name=''):
        path = f"{dir}/{name}.npy"
        np.save(path, rollout_data, allow_pickle=True)

    # Keys that are metadata/settings, not context parameters to pass to the model
    _NON_CONTEXT_KEYS = frozenset({'use_default', 'value_at_margin', 'bounds', 'sigmoid'})

    @staticmethod
    def _flatten_cfg_params(cfg_dict):
        """Recursively flatten a config dict to a sorted list of finite scalar values."""
        if not cfg_dict:
            return []
        values = []
        for k in sorted(cfg_dict.keys()):
            if k in Workspace._NON_CONTEXT_KEYS:
                continue
            v = cfg_dict[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, str):
                continue
            if isinstance(v, (int, float)):
                fv = float(v)
                if np.isfinite(fv):
                    values.append(fv)
            elif isinstance(v, dict):
                values.extend(Workspace._flatten_cfg_params(v))
        return values

    def _get_reward_param(self):
        try:
            params = self._flatten_cfg_params(OmegaConf.to_container(self.cfg.reward_parameters))
        except (omegaconf.errors.ConfigAttributeError, Exception):
            params = []
        return params if params else [0.0]

    def _get_reward_param_dim(self):
        return len(self._get_reward_param())

    def _get_dynamics_param(self):
        try:
            params = self._flatten_cfg_params(OmegaConf.to_container(self.cfg.dynamics_parameters))
        except (omegaconf.errors.ConfigAttributeError, Exception):
            params = []
        return params if params else [0.0]

    def _get_dynamics_param_dim(self):
        return len(self._get_dynamics_param())

    def _get_reward_dynamics_param(self):
        return self._get_reward_param() + self._get_dynamics_param()

    def _get_reward_dynamics_param_dim(self):
        return len(self._get_reward_dynamics_param())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str, default='results')
    parser.add_argument('--eval_mode', choices=['sl_data', 'comparison_data'], default='comparison_data')
    parser.add_argument('--rl_regressor_workdir', type=str, default=None)
    parser.add_argument('--step_to_load', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--rollout_dir', type=str, default='rollout_data')
    parser.add_argument('--video_dir', type=str, default='video_logs')
    args = parser.parse_args()

    cfg_path = Path(args.workdir).joinpath('cfg.yaml')
    cfg = OmegaConf.load(cfg_path)

    workspace = Workspace(cfg, args.workdir, args)

    # Generate the data used for supervised learning
    if args.eval_mode == 'sl_data':
        sl_rollout_fname = f"{workspace.base_name}_rollout"
        rl_rollout_data = workspace.rollout(n_episodes=args.n_episodes, use_approximator=False)
        workspace.save_rollout(rl_rollout_data,
                               dir=workspace.rollout_dir,
                               name=sl_rollout_fname)

    # Rollout RL agent and the approximator
    if args.eval_mode == 'comparison_data':
        assert workspace.rl_regressor is not None, "RL approximator is not loaded."

        # Rollout RL agent if not saved and not MAML
        agent_rollout_fname = f"{workspace.base_name}_rollout_{workspace.base_name}_rollout_agent"
        if not Path(f"{workspace.rollout_comparison_data}/{agent_rollout_fname}.npy").is_file() or workspace.is_meta_learning:
            rl_rollout_data = workspace.rollout(n_episodes=args.n_episodes, use_approximator=False)
        else:
            print("Skipping rolling out the RL agent, because the data is already generated.")

        # Rollout the approximator if not saved
        approx_rollout_fname = f"{workspace.base_name}_rollout_approx-{workspace.rl_regressor_name}_approxseed-{workspace.rl_regressor_seed}"
        if not Path(f"{workspace.rollout_comparison_data}/{approx_rollout_fname}.npy").is_file():
            approximator_rollout_data = workspace.rollout(n_episodes=args.n_episodes, use_approximator=True)
            workspace.save_rollout(approximator_rollout_data,
                                   dir=workspace.rollout_comparison_data,
                                   name=approx_rollout_fname)
        else:
            print(f"Skipping rolling out the {workspace.rl_regressor_name}, because the data is already generated.")

        # Finetune the meta policy with RL rollout and then evaluate it
        if workspace.is_meta_learning:
            finetuned_approx_rollout_fname = f"{workspace.base_name}_rollout_approx-finetuned_{workspace.rl_regressor_name}_approxseed-{workspace.rl_regressor_seed}"
            workspace.finetune_meta_policy(rl_rollout_data)
            finetuned_approximator_rollout_data = workspace.rollout(n_episodes=args.n_episodes, use_approximator=True)
            workspace.save_rollout(finetuned_approximator_rollout_data,
                                   dir=workspace.rollout_comparison_data,
                                   name=finetuned_approx_rollout_fname)

    # Visualization
    if args.vis:
        # Visualization
        plot_type = 'scatter'
        for z_data, label in zip([rl_rollout_data['value'], rl_rollout_data['reward']], ['V(s)', 'R(s)']):
            # visualization of the rollout, values/rewards in the actual MDP
            plots.visualize_phase_space(
                rl_rollout_data['physics_qpos'],
                rl_rollout_data['physics_qvel'],
                z_data, workspace.plot_dir,
                f"{workspace.base_name}_phase_{label}_{plot_type}",
                goal_coord=None,
                plot_type='scatter', label=label
            )


if __name__ == '__main__':
    main()
