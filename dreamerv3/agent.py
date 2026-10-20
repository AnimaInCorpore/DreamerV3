from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence
from .maths import symexp, twohot_loss, expectation, lambda_returns, replay_returns, LaProp
from .networks import RSSM, mlp, categorical


class Agent(nn.Module):
    def __init__(self, obs_dim, actions, cfg):
        super().__init__()
        self.cfg, self.actions = cfg, actions
        self.world = RSSM(obs_dim, actions, cfg['hidden'], cfg['deter'], cfg['latents'], cfg['classes'])
        feat = self.world.feature_dim
        self.actor = mlp(feat, actions, cfg['hidden'])
        self.critic = mlp(feat, 255, cfg['hidden'], zero=True)
        self.slow_critic = deepcopy(self.critic).requires_grad_(False)
        self.register_buffer('bins', symexp(torch.linspace(-20, 20, 255)))
        self.register_buffer('return_low', torch.tensor(0.))
        self.register_buffer('return_high', torch.tensor(0.))
        self.updates = 0
        self.to(cfg['device'])
        self.model_opt = LaProp(self.world.parameters(), lr=cfg['lr'])
        self.actor_opt = LaProp(self.actor.parameters(), lr=cfg['lr'])
        self.critic_opt = LaProp(self.critic.parameters(), lr=cfg['lr'])

    @torch.no_grad()
    def act(self, obs, previous_action, first, state=None, greedy=False):
        state, _, _ = self.world.observe(obs, previous_action, first, state)
        dist = categorical(self.actor(self.world.features(state)))
        action = F.one_hot(dist.probs.argmax(-1), self.actions).float() if greedy else dist.sample()
        return action, state

    def train_world(self, batch):
        from .maths import symlog
        state = (batch['h'][:, 0], batch['z'][:, 0])
        features, priors, posts, hs, zs = [], [], [], [], []
        for t in range(batch['obs'].shape[1]):
            hs.append(state[0].detach())
            zs.append(state[1].detach())
            state, prior, post = self.world.observe(batch['obs'][:, t], batch['action'][:, t],
                                                   batch['first'][:, t], state)
            features.append(self.world.features(state))
            priors.append(prior)
            posts.append(post)
        features = torch.stack(features, 1)
        prior, post = torch.stack(priors, 1), torch.stack(posts, 1)
        dyn = kl_divergence(categorical(post.detach()), categorical(prior)).sum(-1).clamp_min(1)
        rep = kl_divergence(categorical(post), categorical(prior.detach())).sum(-1).clamp_min(1)
        reconstruction = (self.world.decoder(features) - symlog(batch['obs'])).square().sum(-1)
        reward = twohot_loss(self.world.reward(features), batch['reward'], self.bins)
        cont = F.binary_cross_entropy_with_logits(self.world.continue_head(features).squeeze(-1),
                                                  1 - batch['terminal'], reduction='none')
        loss = (reconstruction + reward + cont + dyn + 0.1 * rep).mean()
        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.model_opt.step()
        h, z = features.detach().split([self.world.deter, self.world.stoch], -1)
        metrics = dict(model_loss=loss.detach(), reconstruction=reconstruction.mean().detach(),
                       reward_loss=reward.mean().detach(), dynamics_kl=dyn.mean().detach(),
                       continuation_loss=cont.mean().detach())
        return (h.flatten(0, 1), z.flatten(0, 1)), metrics, torch.stack(hs, 1), torch.stack(zs, 1), features.detach()

    def train_behavior(self, start, terminal, replay_features, batch):
        cfg = self.cfg
        # REINFORCE requires no gradient through the world model or sampled paths.
        with torch.no_grad():
            state = start
            features, actions, rewards, continuations = [self.world.features(state)], [], [], []
            for _ in range(cfg['horizon']):
                action = categorical(self.actor(features[-1])).sample()
                state, _ = self.world.transition(state, action)
                feat = self.world.features(state)
                features.append(feat)
                actions.append(action.argmax(-1))
                rewards.append(expectation(self.world.reward(feat), self.bins))
                continuations.append(self.world.continue_head(feat).sigmoid().squeeze(-1))
            features = torch.stack(features)
            reward, cont = torch.stack(rewards), torch.stack(continuations)
            values = expectation(self.critic(features), self.bins)
            returns = lambda_returns(reward, cont, values[1:], values[-1], cfg['gamma'], cfg['lambda'])
            low, high = torch.quantile(returns, torch.tensor([0.05, 0.95], device=returns.device))
            self.return_low.lerp_(low, 0.01)
            self.return_high.lerp_(high, 0.01)
            scale = (self.return_high - self.return_low).clamp_min(1)
            advantage = (returns - values[:-1]) / scale
            weights = torch.cumprod(torch.cat([(1 - terminal.flatten())[None], cfg['gamma'] * cont[:-1]], 0), 0)
        dist = categorical(self.actor(features[:-1]))
        logp = dist.log_prob(F.one_hot(torch.stack(actions), self.actions).float())
        actor_loss = -(weights * (logp * advantage + cfg['entropy'] * dist.entropy())).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        logits = self.critic(features[:-1])
        with torch.no_grad():
            slow_probs = self.slow_critic(features[:-1]).softmax(-1)
        regularizer = -(slow_probs * logits.log_softmax(-1)).sum(-1)
        critic_loss = (weights * (twohot_loss(logits, returns, self.bins) + regularizer)).mean()
        # Replayed states receive on-policy imagination values as bootstraps.
        with torch.no_grad():
            annotations = returns[0].reshape(batch['reward'].shape)
            replay_return = replay_returns(batch['reward'], batch['terminal'], batch['first'],
                                            annotations, cfg['gamma'], cfg['lambda'])
            valid = 1 - batch['first'][:, 1:]
            replay_slow = self.slow_critic(replay_features[:, :-1]).softmax(-1)
        replay_logits = self.critic(replay_features[:, :-1])
        replay_loss = twohot_loss(replay_logits, replay_return, self.bins)
        replay_reg = -(replay_slow * replay_logits.log_softmax(-1)).sum(-1)
        critic_loss = critic_loss + 0.3 * ((replay_loss + replay_reg) * valid).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for slow, fast in zip(self.slow_critic.parameters(), self.critic.parameters()):
                slow.lerp_(fast, 0.02)
        return dict(actor_loss=actor_loss.detach(), critic_loss=critic_loss.detach(),
                    imagined_reward=reward.mean(), return_scale=scale, entropy=dist.entropy().mean().detach())

    def update(self, batch):
        start, metrics, h, z, features = self.train_world(batch)
        metrics.update(self.train_behavior(start, batch['terminal'], features, batch))
        self.updates += 1
        if not all(torch.isfinite(v).all() for v in metrics.values()):
            raise FloatingPointError('Non-finite training metric')
        return metrics, h, z

    def optimizer_state(self):
        return {k: getattr(self, k).state_dict() for k in ('model_opt', 'actor_opt', 'critic_opt')}

    def load_optimizers(self, states):
        for name, state in states.items():
            getattr(self, name).load_state_dict(state)
