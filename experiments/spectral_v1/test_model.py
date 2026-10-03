import unittest
import torch
from model import SpectralFilter

class TestFilter(unittest.TestCase):
    def test_descent_gradients_and_boundaries(self):
        torch.manual_seed(71);torch.set_num_threads(2)
        def rows(n):
            x=torch.randn(2,n,6);m=torch.rand_like(x)<.3
            return x,m,torch.randn(2,n),torch.randint(0,2,(2,n)).float()
        q,qm,qz,_=rows(19);s,sm,sz,sy=rows(24);t,tm,tz,ty=rows(24)
        inputs=[q,qm,qz,s,sm,sy,sz,t,tm,ty,tz]
        model=SpectralFilter();p=model.prepare(inputs);logits,obj=model.solve(p,True)
        self.assertLessEqual(float((obj[1:]-obj[:-1]).max()),1e-10)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.rand_like(logits));loss.backward()
        self.assertGreater(sum(float(w.grad.abs().sum()) for w in model.network.parameters()),0)
        self.assertTrue(all(torch.isfinite(w.grad).all() for w in model.network.parameters()))
        hidden=[x.clone() for x in inputs]
        for i,j in ((0,1),(3,4),(7,8)):hidden[i][hidden[j]]=float('nan')
        torch.testing.assert_close(model(*inputs),model(*hidden),atol=0,rtol=0)
        split=torch.cat((model(*(x[:,:7] if i<3 else x for i,x in enumerate(inputs))),model(*(x[:,7:] if i<3 else x for i,x in enumerate(inputs)))),1)
        torch.testing.assert_close(logits,split,atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main()
