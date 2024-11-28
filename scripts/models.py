import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import sys
sys.path.append('/home/georgegao/diffusion_policy_ood/ood')

from torch import tensor, log, exp, flatten
from torch.distributions import MultivariateNormal as MVN


class GMMGradient(nn.Module):
    def __init__(self, GMM_PARAMS):
        super(GMMGradient,self).__init__()
        self.GMM_PARAMS = GMM_PARAMS

        # set parameters for negative exponential
        self.eps = 5500
        self.lmb = 1100
    
    def param_negative_exponential(self, x):
        return np.exp((-x+self.eps)/self.lmb)

    def pdfs(self, x):
        densities = []
        x.retain_grad()
        for i in range(len(self.GMM_PARAMS)):
            gmm_params = self.GMM_PARAMS[str(i)][()]
            MVNs = [MVN(mu, sigma) for (mu, sigma) in zip(tensor(gmm_params["means_"]), tensor(gmm_params["covariances_"]))]
            pdf = sum([pi * exp(MVN.log_prob(x[i])) for (pi, MVN) in zip(tensor(gmm_params['weights_']), MVNs)])
            pdf.backward()
            pdf = float(pdf.detach().cpu())
            densities.append([pdf]*3)
        return np.array(densities).flatten()
    
    def forward(self,kps):
        densities = []
        grad_vectors = []
        for i in range(kps.shape[0]):
            dens, grad = self._get_grad(kps[i])
            densities.append(dens)
            grad_vectors.append(grad)
        grad_vectors = np.array(grad_vectors)
        densities = np.array(densities)
        return densities, grad_vectors
    
    def _get_grad(self, z):
        z = tensor(z, requires_grad=True)
        densities = self.pdfs(z)

        grad = z.grad.detach().cpu().numpy()
        negexp_grad = np.copy(grad)
        for i in range(len(self.GMM_PARAMS)):
            grad_kp = grad[i]
            grad_kp_norm = grad_kp/np.linalg.norm(grad_kp)
            mag = self.param_negative_exponential(np.linalg.norm(grad_kp))
            negexp_grad[i] = grad_kp_norm*mag

        return densities, negexp_grad.flatten()
    

