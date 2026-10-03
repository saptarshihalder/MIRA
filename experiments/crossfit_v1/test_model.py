import unittest
import torch
from model import CrossfitOperator

class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(78);torch.set_num_threads(2)
        def rows(n):
            x=torch.randn(2,n,6);m=torch.rand_like(x)<.3
            return x,m,torch.randn(2,n),torch.randint(0,2,(2,n)).float()
        q,qm,qz,_=rows(13);s,sm,sz,sy=rows(24);t,tm,tz,ty=rows(24)
        self.inputs=[q,qm,qz,s,sm,sy,sz,t,tm,ty,tz]
        self.model=CrossfitOperator()

    def test_verification_labels_do_not_fit_candidate(self):
        p=self.model.prepare(self.inputs)[0]
        before=self.model.candidate(p)[0]
        changed={**p,'verification_y':1-p['verification_y']}
        torch.testing.assert_close(before,self.model.candidate(changed)[0],atol=0,rtol=0)
        parts=self.model.prepare(self.inputs);old=self.model.predict(parts)
        altered=[x.clone() for x in self.inputs];altered[0]*=19
        new=self.model.predict(self.model.prepare(altered))
        torch.testing.assert_close(old['gains'],new['gains'],atol=1e-12,rtol=1e-12)

    def test_gradients_inner_descent_and_hidden_values(self):
        output=self.model.predict(self.model.prepare(self.inputs),True)
        y=torch.rand_like(output['soft'])
        loss=sum(torch.nn.functional.binary_cross_entropy(output[k],y) for k in ('soft','unguarded'))/2
        loss.backward()
        for module in (self.model.network,self.model.query_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()))
        for objective in output['objectives']:self.assertLessEqual(float((objective[1:]-objective[:-1]).max().detach()),1e-10)
        hidden=[x.clone() for x in self.inputs]
        for i,j in ((0,1),(3,4),(7,8)):hidden[i][hidden[j]]=float('nan')
        torch.testing.assert_close(self.model(*self.inputs),self.model(*hidden),atol=0,rtol=0)

    def test_exact_frozen_action(self):
        with torch.no_grad():self.model.query_head[-1].weight.zero_();self.model.query_head[-1].bias.fill_(-1e6)
        self.assertTrue(torch.equal(self.model(*self.inputs),self.inputs[2].double()))

if __name__=='__main__':unittest.main()
