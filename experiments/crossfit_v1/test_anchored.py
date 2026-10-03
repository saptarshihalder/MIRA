import unittest
import torch
import run
from anchored_model import AnchoredQuery,AnchoredSupport,AnchorOnly

class AnchorTests(unittest.TestCase):
    def test_projection_boundaries_and_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        inputs=run.v1.tensor_input([run.v1.world(690334,6,run.v1.DEFAULTS,training=True)[0]],'cpu')
        m=AnchoredQuery('query');parts=m.prepare(inputs)
        for p in parts:
            self.assertLess(p['a'].sum(1).abs().max().item(),1e-9)
            self.assertLess((p['a']*p['z'][...,None]).sum(1).abs().max().item(),1e-8)
        changed=[v.clone() for v in inputs];changed[9][:,1::2]=1-changed[9][:,1::2]
        other=m.prepare(changed)[0]
        for k in ('a','q','z','qz','eigen','v'):self.assertTrue(torch.equal(parts[0][k],other[k]))
        for cls in (AnchoredQuery,AnchoredSupport):
            m=cls('query');result=m.predict(parts);result['unguarded'].sum().backward()
            for net in (m.network,m.query_head):self.assertGreater(max(v.grad.abs().max().item() for v in net.parameters()),0)
        result=AnchorOnly('query').predict(parts);self.assertFalse(result['accepted'].any())

if __name__=='__main__':unittest.main()
