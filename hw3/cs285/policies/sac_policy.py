from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from torch import distributions

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device) # log alpha such that it always remains positive(alpha)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        # ???? this is just the alpha ????
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        # below is needed else dimension mismatch error
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]
            
        obs = ptu.from_numpy(obs)
        action_distribution = self(obs)
        if sample:
            action = action_distribution.sample()
        else:
            action = action_distribution.mean
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        # action_distribution = super(MLPPolicySAC,self).forward(observation)
        else:
            batch_mean = self.mean_net(observation)
            # You will need to clip log values
            scale_tril = torch.exp(torch.clamp(self.logstd,self.log_std_bounds[0],self.log_std_bounds[1]))
            
            # You will need SquashedNormal from sac_utils file 
            # We do this so that we are in the interval [-1,1] for output, since this is what the enviroment wants
            # how does it perform if we simply clip the output I wonder?
            action_distribution = sac_utils.SquashedNormal(batch_mean, scale_tril)

            return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        obs = ptu.from_numpy(obs)
        actions_distr = self(obs)
        action = actions_distr.sample()
        
        log_prob = actions_distr.log_prob(action).sum(axis=1) 

        q_1, q_2 = critic(obs, action)
        q_min = torch.min(q_1, q_2)
        
        actor_loss = - q_min.detach() + self.alpha.detach() * log_prob # basically argmax with additional term
        actor_loss = actor_loss.mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


        # now improve the alpha value
        # this might not be correct do some thinking
        # if log_prob is high (0 = no exploration) and target entropy is larger than we are positive (target entropy is negativ)
        # so alpha is decreased (minimization)-> this leads to less exploration => wrong! We want more exploration
        # if log_prob is low (-100 = a lot of exploration) 
        # alpha_loss = (self.alpha * (log_prob.detach()  - self.target_entropy)).mean() 
        
        # this should be correct high log_prob (0) -> low number -> alpha is increased -> more exploration
        # low log_prob (-100) -> high number -> alpha is decreased -> less exploration 
        alpha_loss = (self.alpha * (-log_prob.detach()  + self.target_entropy)).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


        return actor_loss.item(), alpha_loss.item(), self.alpha.detach().item()