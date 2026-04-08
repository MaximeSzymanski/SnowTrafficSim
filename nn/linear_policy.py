import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MontrealLinearModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Dynamically calculate the size of the flattened observation
        # This allows you to change the number of trucks/blowers later without breaking the network
        inner_obs_space = obs_space.original_space["observation"].spaces
        self.flattened_dim = int(
            np.prod(inner_obs_space["intersections"].shape) +
            np.prod(inner_obs_space["blowers"].shape) +
            np.prod(inner_obs_space["trucks"].shape) +
            np.prod(inner_obs_space["edges"].shape)
        )

        # 2. Build the Multi-Layer Perceptron (MLP)
        # 256 neurons is the standard starting point for MARL
        self.core = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 3. Output Branches
        self.logits_branch = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1) # Used by PPO to estimate future rewards
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the observation and mask from the PettingZoo dictionary
        obs = input_dict["obs"]["observation"]
        action_mask = input_dict["obs"]["action_mask"]
        batch_size = action_mask.shape[0]

        # Flatten all the graph data into a single 1D array
        flat_obs = torch.cat([
            obs["intersections"].reshape(batch_size, -1),
            obs["blowers"].reshape(batch_size, -1),
            obs["trucks"].reshape(batch_size, -1),
            obs["edges"].reshape(batch_size, -1)
        ], dim=1)

        # Pass the data through the neural network
        features = self.core(flat_obs)
        logits = self.logits_branch(features)
        
        # Save the value for the PPO critic
        self._cur_value = self.value_branch(features).squeeze(1)

        # --- THE ACTION MASK ---
        # If the mask is 0 (invalid action), subtract 100,000,000 from the neural network's guess.
        # This forces the probability of that action to essentially 0%.
        inf_mask = (1.0 - action_mask) * -1e8
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self._cur_value