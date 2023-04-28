from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure.sac_utils import *
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic 


        # calculate target Q values
        next_ob_no = ptu.from_numpy(next_ob_no)
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        alpha = self.actor.alpha.detach()  # .alpha.detach()

        next_ac_no_distr = self.actor(next_ob_no)
        next_ac_no = next_ac_no_distr.sample()
        q_1, q_2 = self.critic_target(next_ob_no, next_ac_no)
        q_target_min = torch.min(q_1,q_2)
        next_ac_logprob = next_ac_no_distr.log_prob(next_ac_no).detach()
        entropy_reg = q_target_min - alpha * next_ac_logprob
        #print(entropy_reg.shape)
        #print(re_n.shape)
        #print(terminal_n.shape)
        value = re_n + entropy_reg.squeeze() * (1 - terminal_n) * self.gamma
        target = value.detach().unsqueeze(1)
        

        # now update each critic
        qa_t_values = self.critic(ob_no, ac_na)
        # critic 0
        #print(qa_t_values[0].shape)
        #print(target.shape)
        assert qa_t_values[0].shape == target.shape
        critic_loss0 = self.loss(qa_t_values[0], target)

        critic_loss1 = self.loss(qa_t_values[1], target)

        critic_loss = critic_loss0 + critic_loss1

        # optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        critic_loss = 0
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        
        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)
        
        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        actor_loss = 0
        alpha_loss = 0
        temperature = 0
        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                loss = self.actor.update(ob_no, critic=self.critic)
                actor_loss += loss[0]
                alpha_loss += loss[1]
                temperature = loss[2]
                #actor_loss, alpha_loss, temperature += 0,0,0 #self.actor.update(ob_no, critic=self.critic) # TODO!
        
        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = temperature
        print("ji")

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
