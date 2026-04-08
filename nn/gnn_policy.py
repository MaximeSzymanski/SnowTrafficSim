import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MontrealGNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        adj_numpy = kwargs.get("adj_matrix")
        if adj_numpy is None:
            adj_numpy = model_config.get("custom_model_config", {}).get("adj_matrix")
        if adj_numpy is None:
            raise ValueError("GNN Model did not receive the adj_matrix! Check your train_ray.py config.")

        self.num_nodes = len(adj_numpy)
        
        adj_tensor = torch.tensor(adj_numpy, dtype=torch.float32)
        adj_tensor = adj_tensor + torch.eye(self.num_nodes) 
        
        degree = torch.sum(adj_tensor, dim=1)
        d_inv = torch.pow(degree, -1.0) 
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        norm_adj = torch.matmul(d_mat_inv, adj_tensor)
        self.register_buffer("A", norm_adj)

        # Node embed accepts 2 features (Traffic + Snow)
        self.node_embed = nn.Linear(2, 16)
        self.gcn_layer_1 = nn.Linear(16, 32)
        self.gcn_layer_2 = nn.Linear(32, 32)

        # Agent dashboards
        self.blower_embed = nn.Linear(2, 16)
        self.truck_embed = nn.Linear(3, 16)

        # Actor Input is exactly 80.
        # Global Map (32) + Local Intersection Context (32) + Agent Dashboard (16) = 80
        self.actor_net = nn.Sequential(
            nn.Linear(80, 64), nn.ReLU(), nn.Linear(64, num_outputs) 
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"]
        device = obs["intersections"].device
        batch_size = obs["intersections"].shape[0]
        
        # --- 1. GRAPH CONVOLUTION ---
        nodes = obs["intersections"][:, :self.num_nodes, :] 
        
        x = torch.relu(self.node_embed(nodes))
        x = torch.matmul(self.A, x) 
        x = torch.relu(self.gcn_layer_1(x))
        x = torch.matmul(self.A, x) 
        gnn_nodes = torch.relu(self.gcn_layer_2(x)) # Shape: (Batch, Total_Nodes, 32)
        
        # --- 2. EXTRACT LOCAL CONTEXT (DYNAMIC MULTI-AGENT FIX) ---
        
        # Pull the agent index and flatten it. Shape: [Batch]
        agent_idx = obs["agent_index"].long().squeeze(-1) 
        batch_indices = torch.arange(batch_size, device=device)
        
        if self.num_outputs == 5: # Blower Policy
            
            # THE MAGIC: Dynamically slice the exact Blower we care about for this batch
            # my_blower_data Shape: [Batch, 2]
            my_blower_data = obs["blowers"][batch_indices, agent_idx, :]
            
            # Get node ID and extract local GNN embedding
            current_node_idx = my_blower_data[:, 0].long() 
            local_node_embed = gnn_nodes[batch_indices, current_node_idx, :] # [Batch, 32]
            
            # Process the Blower's dashboard stats
            b_scale = torch.tensor([20000.0, 1.0], device=device)
            x_agent = torch.relu(self.blower_embed(my_blower_data / b_scale)) # [Batch, 16]
            
        else: # Truck Policy
            
            # Dynamically slice the exact Truck we care about
            my_truck_data = obs["trucks"][batch_indices, agent_idx, :]
            
            current_node_idx = my_truck_data[:, 0].long()
            local_node_embed = gnn_nodes[batch_indices, current_node_idx, :] # [Batch, 32]
            
            t_scale = torch.tensor([20000.0, 1.0, 10.0], device=device)
            x_agent = torch.relu(self.truck_embed(my_truck_data / t_scale)) # [Batch, 16]

        # --- 3. ASSEMBLE THE BRAIN ---
        # Critic gets the global average to calculate the win condition
        global_map = torch.mean(gnn_nodes, dim=1) # (Batch, 32)
        self._last_value = self.value_head(global_map).squeeze(1)
        
        # Actor gets everything: The city (32), the street corner (32), and the dashboard (16)
        actor_input = torch.cat([global_map, local_node_embed, x_agent], dim=1) # (Batch, 80)
        logits = self.actor_net(actor_input)
        
        # --- 4. ACTION MASKING ---
        action_mask = input_dict["obs"]["action_mask"].bool()
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        
        return masked_logits, state

    def value_function(self):
        return self._last_value