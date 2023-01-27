import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical


class MixtureBernoulliNetwork(nn.Module):
    """
    Mixture density network.

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.bernoulli_network = BernoulliNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.bernoulli_network(x)

    def loss(self, x, y):
        pi, bernoulli = self.forward(x)
        
        mean0 = 1-  bernoulli.probs + 10**(-9)
        mean1 = bernoulli.probs + 10**(-9)
        mean1log = torch.log(mean1 )
        mean0log = torch.log(mean0  )
        
        #full covarinace
        logProbTerm1 = y.unsqueeze(1).expand_as(bernoulli.probs) * mean1log
        logProbTerm2 =  (1-y).unsqueeze(1).expand_as(bernoulli.probs) * mean0log
        
        loglik = logProbTerm1 + logProbTerm2 
        loglik = torch.sum(loglik, dim=2)

        loss = -torch.logsumexp(torch.log(pi.probs+10**(-9) ) + loglik, dim=1) #it is correct, as above

        return loss
    
    def covariance(self,x):
        pi,ber = self.forward(x) #pi.probs : N x K, ber.mean N x K x out-dim 
        responsibility = torch.mean(pi.probs,axis=0)

        # weighted mean: N x K x out-dim 
        weighted_mean = torch.unsqueeze(pi.probs,axis=-1)*ber.mean
        mean_arr = torch.mean(weighted_mean,axis=0) # K x out-dim

        ccov_arr = torch.diag_embed(mean_arr*(1-mean_arr)) # K x dim(covariance) = K x out-dim x out-dim
        mean_prod_arr = torch.matmul(torch.unsqueeze(mean_arr,axis=-1),torch.unsqueeze(mean_arr,axis=1) ) # K x out-dim x out-dim
        comp1_arr = torch.sum(torch.unsqueeze(torch.unsqueeze( responsibility,-1),-1)*(ccov_arr+ mean_prod_arr),axis=0) # out-dim x out-dim
        avemean_arr = torch.sum(torch.unsqueeze(responsibility,axis=-1)*mean_arr,axis=0) # sum_{k=1^K} pi_k * mu_k,shape out-dim
        comp2_arr = torch.matmul(torch.unsqueeze(avemean_arr,dim=-1),torch.unsqueeze(avemean_arr,dim=-1).T)
        
        return comp1_arr - comp2_arr

    def sample(self, x):
        pi, bernoulli = self.forward(x)
        a = pi.sample().unsqueeze(2)
        b = bernoulli.sample()
        samples = torch.sum(a * b , dim=1)

        return samples

    def predict(self,x):
        pi, bernoulli = self.forward(x)
        pred_prob = torch.sum((pi.probs.unsqueeze(1).T * bernoulli.probs.permute(0,2,1).T).T.permute(0,2,1),dim=1)

        return pred_prob
        
    

class BernoulliNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim * n_components),
            nn.Sigmoid(),
        )



    def forward(self, x):
        mean = self.network(x)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        return Bernoulli(probs = mean.transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
           nn.Linear(in_dim, hidden_dim),
           nn.ELU(),
           nn.Linear(hidden_dim, out_dim) ### out_dim, should be n_components.
        )


    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)


