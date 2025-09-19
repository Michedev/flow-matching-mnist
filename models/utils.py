import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def optimal_assignment(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """
    Perform optimal assignment between x_0 and x_1 samples using Hungarian algorithm.
    Returns reordered x_0 to minimize total assignment cost with x_1.
    """
    batch_size = x_0.shape[0]
    x_0_flatten = x_0.flatten(1)  # [B, num_features]
    x_1_flatten = x_1.flatten(1)  # [B, num_features]

    # Compute pairwise distances: [B, B]
    distance_matrix = torch.cdist(x_0_flatten, x_1_flatten, p=2)
    
    # Convert to numpy for Hungarian algorithm
    cost_matrix = distance_matrix.cpu().numpy()
    
    # Find optimal assignment using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create reordered x_0 based on optimal assignment
    x_0_reordered = x_0.clone()
    for i, j in zip(row_indices, col_indices):
        x_0_reordered[i] = x_0[j]

    return x_0_reordered

