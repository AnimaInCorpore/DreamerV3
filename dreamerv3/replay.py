import numpy as np
import torch


class Replay:
    """Per-environment ring sequences, including explicit reset and terminal flags.

    Entries pair observation[t] with the preceding action/reward/terminal. Latent
    states are captured by the actor and refreshed after model training. The first
    sequence in each batch is the most recently collected non-overlapping chunk.
    """
    def __init__(self, capacity, envs, obs_dim, actions, deter, stoch, seed):
        self.capacity = max(2, capacity // envs)
        self.envs, self.pos, self.count = envs, 0, 0
        self.rng = np.random.default_rng(seed)
        shapes = dict(obs=(obs_dim,), action=(actions,), reward=(), first=(), terminal=(),
                      h=(deter,), z=(stoch,))
        self.data = {k: np.zeros((self.capacity, envs, *v), np.float32) for k, v in shapes.items()}
        self.online_count = 0

    def add(self, **entry):
        for key, value in entry.items():
            self.data[key][self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.online_count += 1

    def sample(self, batch, length, device):
        if self.count < length:
            raise ValueError('Replay does not yet contain a complete sequence')
        starts = self.rng.integers(0, self.count - length + 1, batch)
        env = self.rng.integers(0, self.envs, batch)
        if self.online_count >= length:
            starts[0] = self.count - length
            self.online_count = 0
        oldest = self.pos if self.count == self.capacity else 0
        rows = (starts[:, None] + np.arange(length)[None, :] + oldest) % self.capacity
        out = {k: torch.as_tensor(v[rows, env[:, None]], device=device) for k, v in self.data.items()}
        return out, (rows, env)

    def refresh(self, indices, h, z):
        rows, env = indices
        self.data['h'][rows, env[:, None]] = h.detach().cpu().numpy()
        self.data['z'][rows, env[:, None]] = z.detach().cpu().numpy()

    def state_dict(self):
        # Only serialize occupied entries, not the unused multi-GB capacity.
        return dict(capacity=self.capacity, envs=self.envs, pos=self.pos, count=self.count,
                    online_count=self.online_count, rng=self.rng.bit_generator.state,
                    data={k: v[:self.count].copy() for k, v in self.data.items()})

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity or state['envs'] != self.envs:
            raise ValueError('Replay checkpoint capacity or environment count mismatch')
        self.pos, self.count = state['pos'], state['count']
        self.online_count = state['online_count']
        self.rng.bit_generator.state = state['rng']
        for k, v in state['data'].items():
            self.data[k][:self.count] = v
