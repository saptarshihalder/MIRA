"""Numerical-only repair: convex ridge calibration via certified Newton steps."""
import numpy as np
from scipy.special import expit
import large_native as original


def calibration(z,labels,kind='platt',ridge=.001):
    z=np.asarray(z,dtype=float)
    labels=np.asarray(labels,dtype=float)
    design=np.ones((len(z),1)) if kind=='intercept' else np.column_stack((z,np.ones(len(z))))
    theta=np.zeros(design.shape[1])
    def quantities(value):
        score=z+design@value
        probability=expit(score)
        loss=float(np.mean(np.logaddexp(0.,score)-labels*score)+.5*ridge*(value@value))
        gradient=design.T@(probability-labels)/len(labels)+ridge*value
        curvature=probability*(1-probability)
        hessian=(design.T*curvature)@design/len(labels)+ridge*np.eye(len(value))
        return loss,gradient,hessian
    for iteration in range(100):
        loss,gradient,hessian=quantities(theta)
        if np.max(np.abs(gradient))<=1e-8:
            break
        direction=np.linalg.solve(hessian,gradient)
        scale=1.
        for backtrack in range(60):
            candidate=theta-scale*direction
            updated=quantities(candidate)[0]
            if updated<=loss-1e-4*scale*(gradient@direction)+1e-14:
                theta=candidate
                break
            scale*=.5
        else:
            raise RuntimeError('Calibration Newton line search failed')
    loss,gradient,_=quantities(theta)
    error=float(np.max(np.abs(gradient)))
    if error>1e-7:
        raise RuntimeError('Calibration did not attain declared stationarity: '+str(error))
    coefficient=np.array([1.,theta[0]]) if kind=='intercept' else np.array([1.+theta[0],theta[1]])
    return coefficient,dict(success=True,message='Convex ridge Newton stationarity verified',
                            iterations=iteration+1,gradient_max=error,objective=loss,ridge=float(ridge),
                            tolerance=1e-7,algorithm='Newton with Armijo backtracking; same objective and ridge')


if __name__=='__main__':
    original.calibration=calibration
    original.main()
