import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from dreamerv3.maths import symlog, symexp, twohot_loss, lambda_returns, replay_returns, LaProp
from dreamerv3.networks import RSSM
from dreamerv3.replay import Replay
from dreamerv3.agent import Agent
from dreamerv3.envs import PointMass, make_env


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_symlog_roundtrip(self):
        x = torch.tensor([-1e6, -1., 0., 1., 1e6])
        torch.testing.assert_close(symexp(symlog(x)), x)

    def test_external_environment_factory_contract(self):
        env = make_env('dreamerv3.envs:PointMass')
        obs,_ = env.reset(seed=10)
        self.assertEqual(obs.shape,(3,))
        self.assertEqual(len(env.step(1)),5)
        env.close()

    def test_twohot_interpolates_and_clips(self):
        logits = torch.tensor([[1., 2., 3.]]).expand(5, -1).clone().requires_grad_()
        bins = torch.tensor([-2., 0., 2.])
        y = torch.tensor([-9., -1., 0., 1., 9.])
        loss = twohot_loss(logits, y, bins)
        lp = logits.log_softmax(-1)
        expected = -torch.stack([lp[0,0], (lp[1,0]+lp[1,1])/2, lp[2,1],
                                 (lp[3,1]+lp[3,2])/2, lp[4,2]])
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_terminal_stops_lambda_bootstrap(self):
        r = torch.tensor([[1.], [2.], [100.]])
        c = torch.tensor([[1.], [0.], [1.]])
        result = lambda_returns(r, c, torch.ones_like(r)*99, torch.tensor([1000.]), 1., 1.)
        torch.testing.assert_close(result[:2,0], torch.tensor([3.,2.]))

    def test_time_limit_bootstraps_before_reset_without_episode_leakage(self):
        reward = torch.tensor([[0., 1., 0., 1000.]])
        terminal = torch.zeros_like(reward)
        first = torch.tensor([[0., 0., 1., 0.]])
        annotation = torch.tensor([[2., 5., 7., 9.]])
        result = replay_returns(reward,terminal,first,annotation,gamma=1.,lam=1.)
        # Transition 0->1 has reward 1 and bootstraps from final state value 5.
        torch.testing.assert_close(result[0,0],torch.tensor(6.))
        terminal[0,1] = 1
        result = replay_returns(reward,terminal,first,annotation,gamma=1.,lam=1.)
        torch.testing.assert_close(result[0,0],torch.tensor(1.))

    def test_episode_reset_discards_previous_state_and_action(self):
        world = RSSM(3, 3, hidden=16, deter=16, latents=2, classes=4)
        obs = torch.randn(2,3)
        state = (torch.randn(2,16),torch.randn(2,8))
        previous_action = torch.randn(2,3)
        torch.manual_seed(100)
        a, prior_a, post_a = world.observe(obs, previous_action, torch.ones(2), state)
        torch.manual_seed(100)
        b, prior_b, post_b = world.observe(obs, torch.zeros(2,3), torch.ones(2))
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y)
        torch.testing.assert_close(prior_a,prior_b)
        torch.testing.assert_close(post_a,post_b)

    def test_ring_sequences_stay_contiguous_after_wrap_and_restore(self):
        replay = Replay(20, 2, 1, 2, 8, 4, 0)
        for t in range(27):
            replay.add(obs=np.array([[t],[1000+t]]), action=np.zeros((2,2)),
                       reward=np.zeros(2), first=np.zeros(2), terminal=np.zeros(2),
                       h=np.zeros((2,8)),z=np.zeros((2,4)))
        batch, _ = replay.sample(64, 5, 'cpu')
        torch.testing.assert_close(batch['obs'][:,1:,0]-batch['obs'][:,:-1,0],torch.ones(64,4))
        self.assertTrue((batch['obs'][:,:,0] % 1000 >= 17).all())
        restored = Replay(20,2,1,2,8,4,2)
        restored.load_state_dict(replay.state_dict())
        a,_ = replay.sample(4,5,'cpu')
        b,_ = restored.sample(4,5,'cpu')
        torch.testing.assert_close(a['obs'],b['obs'])

    def test_laprop_normalizes_before_momentum_and_restores(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = LaProp([p],lr=.01)
        p.grad = torch.tensor([.1])
        opt.step()
        torch.testing.assert_close(p,torch.tensor([.99]))
        state = copy.deepcopy(opt.state_dict())
        q = torch.nn.Parameter(p.detach().clone())
        opt2 = LaProp([q],lr=.01)
        opt2.load_state_dict(state)
        p.grad, q.grad = torch.tensor([-.2]),torch.tensor([-.2])
        opt.step()
        opt2.step()
        torch.testing.assert_close(p,q)

    def test_full_update_changes_all_three_networks_and_checkpoint_roundtrip(self):
        cfg=dict(hidden=16,deter=16,latents=2,classes=4,device='cpu',lr=3e-4,
                 horizon=3,gamma=.997,**{'lambda':.95},entropy=3e-4)
        agent = Agent(3,3,cfg)
        replay = Replay(100,2,3,3,16,8,0)
        envs = [PointMass(),PointMass()]
        for t in range(12):
            obs = np.stack([e.reset(seed=t+i)[0] for i,e in enumerate(envs)])
            replay.add(obs=obs,action=np.eye(3,dtype=np.float32)[[0,2]],
                       reward=np.array([.5,.8]),first=np.full(2,t==0),terminal=np.zeros(2),
                       h=np.zeros((2,16)),z=np.zeros((2,8)))
        before = {name:copy.deepcopy(net.state_dict()) for name,net in
                  [('world',agent.world),('actor',agent.actor),('critic',agent.critic)]}
        batch, indices = replay.sample(2,4,'cpu')
        metrics,h,z = agent.update(batch)
        replay.refresh(indices,h,z)
        self.assertTrue(all(torch.isfinite(x) for x in metrics.values()))
        for name, old in before.items():
            self.assertTrue(any(not torch.equal(old[k],v) for k,v in getattr(agent,name).state_dict().items()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'agent.pt'
            torch.save(dict(model=agent.state_dict(),optim=agent.optimizer_state()),path)
            restored = Agent(3,3,cfg)
            saved = torch.load(path,weights_only=False)
            restored.load_state_dict(saved['model'])
            restored.load_optimizers(saved['optim'])
            for k,v in agent.state_dict().items():
                torch.testing.assert_close(v,restored.state_dict()[k])
            self.assertEqual(len(agent.model_opt.state),len(restored.model_opt.state))


if __name__ == '__main__':
    unittest.main()
