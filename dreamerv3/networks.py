import torch
from torch import nn
from torch.distributions import OneHotCategoricalStraightThrough


def mlp(inp, out, hidden, layers=3, zero=False):
    modules = []
    for _ in range(layers):
        modules += [nn.Linear(inp, hidden), nn.RMSNorm(hidden), nn.SiLU()]
        inp = hidden
    final = nn.Linear(inp, out)
    if zero:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    return nn.Sequential(*modules, final)


def categorical(logits):
    probs = 0.99 * logits.softmax(-1) + 0.01 / logits.shape[-1]
    return OneHotCategoricalStraightThrough(probs=probs)


class BlockGRU(nn.Module):
    def __init__(self, inp, deter, hidden, blocks=8):
        super().__init__()
        assert deter % blocks == 0
        self.blocks, self.width = blocks, deter // blocks
        self.mix = mlp(inp + deter, hidden, hidden, layers=1)
        self.input = nn.Linear(hidden, 3 * deter)
        self.recurrent = nn.Parameter(torch.empty(blocks, self.width, 3 * self.width))
        nn.init.xavier_uniform_(self.recurrent)
        self.norm = nn.RMSNorm(3 * self.width)

    def forward(self, x, h):
        mixed = self.mix(torch.cat([x, h], -1))
        rec = torch.einsum('nbi,bij->nbj', h.reshape(-1, self.blocks, self.width), self.recurrent)
        inp = self.norm(self.input(mixed).reshape(-1, self.blocks, 3 * self.width))
        ir, ic, iu = inp.chunk(3, -1)
        rr, rc, ru = rec.chunk(3, -1)
        reset = torch.sigmoid(ir + rr)
        candidate = torch.tanh(ic + reset * rc)
        update = torch.sigmoid(iu + ru - 1)
        return (update * candidate + (1 - update) * h.reshape(-1, self.blocks, self.width)).flatten(1)


class RSSM(nn.Module):
    def __init__(self, obs_dim, actions, hidden=128, deter=256, latents=16, classes=16):
        super().__init__()
        self.deter, self.latents, self.classes = deter, latents, classes
        self.stoch = latents * classes
        self.encoder = mlp(obs_dim, hidden, hidden)
        self.gru = BlockGRU(self.stoch + actions, deter, hidden)
        self.prior = mlp(deter, self.stoch, hidden, layers=1)
        self.posterior = mlp(deter + hidden, self.stoch, hidden, layers=1)
        self.feature_dim = deter + self.stoch
        self.decoder = mlp(self.feature_dim, obs_dim, hidden)
        self.reward = mlp(self.feature_dim, 255, hidden, layers=1, zero=True)
        self.continue_head = mlp(self.feature_dim, 1, hidden, layers=1)

    def initial(self, batch, device):
        return (torch.zeros(batch, self.deter, device=device),
                torch.zeros(batch, self.stoch, device=device))

    def logits(self, x):
        return x.reshape(*x.shape[:-1], self.latents, self.classes)

    def transition(self, state, action):
        h, z = state
        h = self.gru(torch.cat([z, action], -1), h)
        prior = self.logits(self.prior(h))
        z = categorical(prior).rsample().flatten(-2)
        return (h, z), prior

    def observe(self, obs, action, first, state=None):
        from .maths import symlog
        if state is None:
            state = self.initial(len(obs), obs.device)
        mask = (1 - first.float()).unsqueeze(-1)
        state = tuple(s * mask for s in state)
        action = action * mask
        (h, _), prior = self.transition(state, action)
        post = self.logits(self.posterior(torch.cat([h, self.encoder(symlog(obs))], -1)))
        z = categorical(post).rsample().flatten(-2)
        return (h, z), prior, post

    @staticmethod
    def features(state):
        return torch.cat(state, -1)
