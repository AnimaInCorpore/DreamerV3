import gymnasium as gym
import importlib
import numpy as np
from gymnasium import spaces


class PointMass(gym.Env):
    """A damped mass must reach a random goal using left/coast/right thrust.

    Continuous physical state, discrete controls, finite-horizon truncation.
    Reward has its maximum at the goal with zero velocity. No demonstrations.
    """
    observation_space = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.np_random.uniform(-1, 1)
        self.v = self.np_random.uniform(-0.2, 0.2)
        self.goal = self.np_random.uniform(-0.75, 0.75)
        self.steps = 0
        return self.obs(), {}

    def obs(self):
        return np.array([self.x, self.v, self.goal], np.float32)

    def step(self, action):
        self.v = 0.95 * self.v + 0.12 * (int(action) - 1)
        self.x = np.clip(self.x + 0.1 * self.v, -3, 3)
        self.steps += 1
        reward = 1 - (self.x - self.goal) ** 2 - 0.1 * self.v ** 2
        return self.obs(), float(reward), False, self.steps >= 100, {}


def make_env(task):
    if task == 'pointmass':
        return PointMass()
    if task == 'cartpole':
        return gym.make('CartPole-v1')
    if ':' in task:
        module, factory = task.rsplit(':', 1)
        env = getattr(importlib.import_module(module), factory)()
        if not isinstance(env.observation_space, spaces.Box) or len(env.observation_space.shape) != 1:
            env.close()
            raise TypeError('Dreamer currently requires a one-dimensional Box observation space')
        if not isinstance(env.action_space, spaces.Discrete):
            env.close()
            raise TypeError('Dreamer currently requires Discrete actions')
        return env
    raise ValueError(f'Unknown task: {task}')
