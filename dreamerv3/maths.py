import torch
import torch.nn.functional as F


def symlog(x):
    return x.sign() * x.abs().log1p()


def symexp(x):
    return x.sign() * x.abs().expm1()


def twohot_loss(logits, target, bins):
    # Interpolate in the original value space between exponentially spaced bins.
    target = target.clamp(bins[0], bins[-1])
    hi = torch.searchsorted(bins, target.contiguous()).clamp(1, len(bins) - 1)
    lo = hi - 1
    weight = (target - bins[lo]) / (bins[hi] - bins[lo])
    logp = logits.log_softmax(-1)
    return -((1 - weight) * logp.gather(-1, lo[..., None]).squeeze(-1)
             + weight * logp.gather(-1, hi[..., None]).squeeze(-1))


def expectation(logits, bins):
    # The two tails can each sum to millions while their difference is < 1.
    # Accumulate in float64 to retain that small difference on native CUDA.
    p = logits.softmax(-1).double()
    bins = bins.double()
    mid = len(bins) // 2
    # Accumulate small to large separately on either side of zero.
    neg = (p[..., :mid] * bins[:mid]).flip(-1).sum(-1)
    pos = (p[..., mid:] * bins[mid:]).sum(-1)
    return (neg + pos).to(logits.dtype)


def lambda_returns(reward, continuation, value, bootstrap, gamma=0.997, lam=0.95):
    result, last = [], bootstrap
    for t in reversed(range(reward.shape[0])):
        last = reward[t] + gamma * continuation[t] * ((1 - lam) * value[t] + lam * last)
        result.append(last)
    return torch.stack(result[::-1])


def replay_returns(reward, terminal, first, annotation, gamma=0.997, lam=0.95):
    """Batch-major returns, preserving truncation bootstraps across reset records.

    Rewards and terminal flags describe the transition into each observation.
    A reset record is not an environment transition: bootstrap at the preceding
    final observation instead of leaking the next episode into this episode.
    """
    result, last = [], annotation[:, -1]
    for t in reversed(range(reward.shape[1] - 1)):
        target = reward[:, t+1] + gamma * (1 - terminal[:, t+1]) * (
            (1 - lam) * annotation[:, t+1] + lam * last)
        last = torch.where(first[:, t+1].bool(), annotation[:, t], target)
        result.append(last)
    return torch.stack(result[::-1], 1)


class LaProp(torch.optim.Optimizer):
    """RMS-normalize gradients before momentum; bias-correct both moments."""
    def __init__(self, params, lr=4e-5, betas=(0.9, 0.99), eps=1e-20, agc=0.3):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, agc=agc))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                limit = group['agc'] * p.norm().clamp_min(1e-3)
                g = g * (limit / g.norm().clamp_min(1e-6)).clamp_max(1)
                state = self.state[p]
                if not state:
                    state.update(step=0, v=torch.zeros_like(p), m=torch.zeros_like(p))
                state['step'] += 1
                state['v'].lerp_(g.square(), 1 - b2)
                normalized = g / ((state['v'] / (1 - b2 ** state['step'])).sqrt() + group['eps'])
                state['m'].lerp_(normalized, 1 - b1)
                p.add_(state['m'], alpha=-group['lr'] / (1 - b1 ** state['step']))
