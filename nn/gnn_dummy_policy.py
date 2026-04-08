import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MontrealGNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Create actual layers with weights so PyTorch can calculate gradients
        self.actor = nn.Linear(1, num_outputs)
        self.critic = nn.Linear(1, 1)
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]["action_mask"].shape[0]
        device = input_dict["obs"]["action_mask"].device
        
        # Create a basic input tensor of shape [batch_size, 1] 
        dummy_input = torch.ones((batch_size, 1), device=device)
        
        # Pass it through the layers. Now PyTorch has a gradient path!
        logits = self.actor(dummy_input)
        
        # Apply the action mask so we don't pick illegal moves
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = logits + inf_mask

        # Calculate the Critic value
        self._last_value = self.critic(dummy_input).squeeze(1)
        
        return masked_logits, state

    def value_function(self):
        return self._last_value