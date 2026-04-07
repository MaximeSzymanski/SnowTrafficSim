import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MontrealGNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.intersection_embed = nn.Linear(1, 32)
        self.blower_embed = nn.Linear(2, 32)
        self.truck_embed = nn.Linear(3, 32)

        self.actor_net = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, num_outputs)
        )
        self.value_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"]
        action_mask = input_dict["obs"]["action_mask"]

        # 1. Node Embeddings
        x_inter = self.intersection_embed(obs["intersections"])
        x_blower = self.blower_embed(obs["blowers"])
        x_truck = self.truck_embed(obs["trucks"])

        # 2. Latent selection (Blower=5, Truck=others)
        latent = torch.mean(x_blower if self.num_outputs == 5 else x_truck, dim=1)
        
        # 3. Actor Output
        logits = self.actor_net(latent)
        
        # 4. Masking: Standard basic log-mask
        inf_mask = torch.clamp(torch.log(action_mask.float()), min=-1e10)
        masked_logits = logits + inf_mask

        # 5. Critic: Simple mean over map state
        global_map = torch.mean(x_inter, dim=1)
        self._last_value = self.value_head(global_map).squeeze(1)
        
        return masked_logits, state

    def value_function(self):
        return self._last_value