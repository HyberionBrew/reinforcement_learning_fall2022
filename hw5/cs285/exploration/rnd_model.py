from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import copy

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        assert self.n_layers >= 1, "n_layers must be at least 1"
        layers = []
        layers.append(nn.Linear(self.ob_dim, self.size))
        layers.append(nn.ReLU(inplace=True))
        # Hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(size, size))
            layers.append(nn.ReLU(inplace=True))
        # Output layer
        layers.append(nn.Linear(size, self.output_size))
        self.f = nn.Sequential(*layers)
        self.f_hat = copy.deepcopy(self.f)
        init_method_1(self.f)
        init_method_2(self.f_hat)
        self.optimizer = self.optimizer_spec.constructor(self.f_hat.parameters(), **self.optimizer_spec.optim_kwargs)
        # self.scheduler = self.optimizer_spec.scheduler_constructor(optimizer, **self.optimizer_spec.scheduler_kwargs)
        print(optimizer_spec)
        
        

    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        error = torch.mean(self.f_hat(ob_no) - self.f(ob_no).detach(), dim=0)
        return error
        

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        assert self.t is not None, "t must be initialized"
        current_lr = self.optimizer_spec.learning_rate_schedule(self.t)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        loss = self(ob_no)

        # get new learning rate 

        # backward pass
        self.optimizer_spec.optimizer.zero_grad()
        loss.backward()
        self.optimizer_spec.optimizer.step()
        #self.t += 1
        return loss.item()