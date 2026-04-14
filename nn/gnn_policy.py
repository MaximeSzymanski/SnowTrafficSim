import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MontrealGNNModel(TorchModelV2, nn.Module):
    """
    A Graph Neural Network (GNN) policy model for a multi-agent reinforcement learning 
    environment. Utilizes sparse matrix operations to efficiently process city-scale 
    graph data and supports dynamic action spaces for heterogeneous agent types 
    (SnowBlowers and DumpTrucks).
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        """
        Initializes the model architecture, processes the adjacency matrix into 
        a sparse tensor, and defines the GCN, actor, and critic networks.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        adj_numpy = kwargs.get("adj_matrix")
        if adj_numpy is None:
            adj_numpy = model_config.get("custom_model_config", {}).get("adj_matrix")
        if adj_numpy is None:
            raise ValueError("GNN Model requires an 'adj_matrix' in custom_model_config.")

        self.num_nodes = len(adj_numpy)
        
        # Precompute the normalized sparse adjacency matrix with self-loops
        adj_tensor = torch.tensor(adj_numpy, dtype=torch.float32)
        adj_tensor = adj_tensor + torch.eye(self.num_nodes) 
        
        degree = torch.sum(adj_tensor, dim=1)
        d_inv = torch.pow(degree, -1.0) 
        d_inv[torch.isinf(d_inv)] = 0.
        
        norm_adj = adj_tensor * d_inv.unsqueeze(1) 
        self.register_buffer("A_sparse", norm_adj.to_sparse(), persistent=False)

        # Graph Convolutional Network layers
        self.node_embed = nn.Linear(2, 16)
        self.gcn_layer_1 = nn.Linear(16, 32)
        self.gcn_layer_2 = nn.Linear(32, 32)

        # Agent-specific telemetry embeddings
        self.blower_embed = nn.Linear(2, 16)
        self.truck_embed = nn.Linear(3, 16)

        # Actor-Critic heads
        self.actor_net = nn.Sequential(
            nn.Linear(80, 64), 
            nn.ReLU(), 
            nn.Linear(64, num_outputs) 
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)
        )
        self._last_value = None

    def _sparse_batch_mm(self, sparse_matrix, dense_batch):
        """
        Performs batched sparse-dense matrix multiplication.
        
        Args:
            sparse_matrix (torch.sparse.Tensor): The adjacency matrix of shape (N, N).
            dense_batch (torch.Tensor): The batched node features of shape (B, N, F).
            
        Returns:
            torch.Tensor: The updated node features of shape (B, N, F).
        """
        B, N, F = dense_batch.shape
        reshaped_batch = dense_batch.permute(1, 0, 2).reshape(N, B * F)
        out = torch.sparse.mm(sparse_matrix, reshaped_batch)
        return out.reshape(N, B, F).permute(1, 0, 2)

    def forward(self, input_dict, state, seq_lens):
        """
        Executes the forward pass of the model, performing message passing across 
        the graph, extracting localized agent features, and outputting masked logits.
        """
        obs = input_dict["obs"]["observation"]
        device = obs["intersections"].device
        batch_size = obs["intersections"].shape[0]
        
        nodes = obs["intersections"][:, :self.num_nodes, :] 
        
        # Execute message passing through the graph
        x = torch.relu(self.node_embed(nodes))
        
        x = self._sparse_batch_mm(self.A_sparse, x) 
        x = torch.relu(self.gcn_layer_1(x))
        
        x = self._sparse_batch_mm(self.A_sparse, x) 
        gnn_nodes = torch.relu(self.gcn_layer_2(x)) 
        
        agent_idx = obs["agent_index"].long().squeeze(-1) 
        batch_indices = torch.arange(batch_size, device=device)
        
        # Dynamically slice features based on the active agent's policy type
        if self.num_outputs == 5: 
            my_blower_data = obs["blowers"][batch_indices, agent_idx, :]
            current_node_idx = my_blower_data[:, 0].long() 
            local_node_embed = gnn_nodes[batch_indices, current_node_idx, :] 
            
            b_scale = torch.tensor([20000.0, 1.0], device=device)
            x_agent = torch.relu(self.blower_embed(my_blower_data / b_scale)) 
            
        else: 
            my_truck_data = obs["trucks"][batch_indices, agent_idx, :]
            current_node_idx = my_truck_data[:, 0].long()
            local_node_embed = gnn_nodes[batch_indices, current_node_idx, :] 
            
            t_scale = torch.tensor([20000.0, 1.0, 10.0], device=device)
            x_agent = torch.relu(self.truck_embed(my_truck_data / t_scale)) 

        # Aggregate global state for the Critic
        global_map = torch.mean(gnn_nodes, dim=1) 
        self._last_value = self.value_head(global_map).squeeze(1)
        
        # Concatenate macro, micro, and telemetry data for the Actor
        actor_input = torch.cat([global_map, local_node_embed, x_agent], dim=1) 
        logits = self.actor_net(actor_input)
        
        # Apply environment-provided action masking
        action_mask = input_dict["obs"]["action_mask"].bool()
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        
        return masked_logits, state

    def value_function(self):
        """
        Returns the value prediction for the current state.
        """
        return self._last_value