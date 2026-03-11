"""
Shared env types for Brax and training loop. No dm_env dependency.
"""

from typing import Any, NamedTuple


# Step type constants (same as dm_env.StepType)
STEP_FIRST = 0
STEP_MID = 1
STEP_LAST = 2


class TimeStep(NamedTuple):
    """One env step: observation, reward, discount, step_type."""
    observation: Any
    reward: Any
    discount: Any
    step_type: int

    def first(self):
        return self.step_type == STEP_FIRST

    def mid(self):
        return self.step_type == STEP_MID

    def last(self):
        return self.step_type == STEP_LAST


class ArraySpec(NamedTuple):
    """Spec for array (observation, reward, discount)."""
    shape: tuple
    dtype: Any
    name: str


class BoundedArraySpec(NamedTuple):
    """Spec for bounded array (action)."""
    shape: tuple
    dtype: Any
    minimum: float
    maximum: float
    name: str


class ExtendedTimeStep(NamedTuple):
    """Time step with action field for replay buffer compatibility."""
    step_type: int
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == STEP_FIRST

    def mid(self):
        return self.step_type == STEP_MID

    def last(self):
        return self.step_type == STEP_LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper:
    """Wraps an env so each time step includes an action (for replay storage)."""

    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        import numpy as np
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=float(time_step.reward) if time_step.reward is not None else 0.0,
            discount=float(time_step.discount) if time_step.discount is not None else 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
