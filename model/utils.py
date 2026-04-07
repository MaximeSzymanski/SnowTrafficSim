import torch
from torch_geometric.data import HeteroData

def obs_to_pyg(obs_dict, device="cpu"):
    """
    Converts the dictionary observation from MontrealSnowEnv 
    into a PyTorch Geometric HeteroData object.
    """
    data = HeteroData()

    # 1. NODE FEATURES
    # Intersections: [Lock_Status]
    data['intersection'].x = torch.tensor(obs_dict["intersections"], dtype=torch.float, device=device)
    
    # Blowers: [Node_ID, Is_Waiting]
    # Note: We strip the Node_ID for the feature vector and keep it for connectivity
    data['blower'].x = torch.tensor(obs_dict["blowers"][:, 1:], dtype=torch.float, device=device)
    
    # Trucks: [Node_ID, Payload, Master_ID]
    data['truck'].x = torch.tensor(obs_dict["trucks"][:, 1:], dtype=torch.float, device=device)

    # 2. CONNECTIVITY (Edges)
    # We need the street network (Intersection -> Intersection)
    # This is usually passed as a static edge_index during reset 
    # but we can re-verify it here.
    # data['intersection', 'street', 'intersection'].edge_index = ...
    
    # Dynamic Connectivity: Where are the vehicles?
    # Blower 'at' Intersection
    blower_nodes = obs_dict["blowers"][:, 0].astype(int)
    blower_indices = torch.arange(len(blower_nodes), device=device)
    data['blower', 'at', 'intersection'].edge_index = torch.stack([
        blower_indices, 
        torch.tensor(blower_nodes, device=device)
    ], dim=0)

    # Truck 'at' Intersection
    truck_nodes = obs_dict["trucks"][:, 0].astype(int)
    truck_indices = torch.arange(len(truck_nodes), device=device)
    data['truck', 'at', 'intersection'].edge_index = torch.stack([
        truck_indices, 
        torch.tensor(truck_nodes, device=device)
    ], dim=0)

    # 3. EDGE FEATURES
    # The snow depth on the streets
    data['intersection', 'street', 'intersection'].edge_attr = torch.tensor(
        obs_dict["edges"], dtype=torch.float, device=device
    )

    return data