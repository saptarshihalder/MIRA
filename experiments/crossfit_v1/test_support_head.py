import unittest
import torch
import run
from support_head import SupportHead

class SupportOnlyTests(unittest.TestCase):
    def test_boundary_capacity_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(9)
        m=SupportHead('query')
        e=run.v1.world(690333,6,run.v1.DEFAULTS,training=True)[0]
        parts=m.prepare(run.v1.tensor_input([e],'cpu'));p=parts[0]
        contexts=[]
        hook=m.query_head.register_forward_pre_hook(lambda module,args:contexts.append(args[0].detach().clone()))
        m.candidate(p)
        changed=dict(p,q=p['q']+12,qz=p['qz']-7,verification_y=1-p['verification_y'])
        m.candidate(changed);hook.remove()
        self.assertTrue(torch.equal(contexts[0],contexts[1]))
        self.assertEqual(sum(v.numel() for v in m.parameters() if v.requires_grad),370)
        result=m.predict(parts);result['unguarded'].sum().backward()
        for net in (m.network,m.query_head):
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in net.parameters()))
            self.assertGreater(max(v.grad.abs().max().item() for v in net.parameters()),0)

if __name__=='__main__':unittest.main()
