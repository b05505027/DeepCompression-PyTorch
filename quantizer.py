import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import numpy as np
import time

class Quantizer:
    def __init__(self, model):
        self.model = model
        self.index_matrices = {}  # To store index matrix for each module
    
    def load_index_matrices(self, folder='default'):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                self.index_matrices[name] = torch.load(f'{folder}/index_matrices/' + name + '.pth')

    def quantize_weights(self, folder='default', conv_bit=8, fc_bit=5):
        for name, module in tqdm(self.model.named_modules(), desc='Quantizing weights', total=len(list(self.model.named_modules()))):
            if isinstance(module, torch.nn.Conv2d):
                self.index_matrices[name] = self._create_index_matrix(module.weight, n_clusters=np.power(2, conv_bit))
            elif isinstance(module, torch.nn.Linear):
                self.index_matrices[name] = self._create_index_matrix(module.weight, n_clusters=np.power(2, fc_bit))
            else:
                continue
            # save index matrix, which contains tensor of indices
            os.makedirs(f'{folder}/index_matrices', exist_ok=True)
            torch.save(self.index_matrices[name], f'{folder}/index_matrices/' + name + '.pth')




    def _create_index_matrix(self, weight_tensor, n_clusters):
        original_shape = weight_tensor.shape
        flattened_weights = weight_tensor.detach().view(-1, 1)
        
        # Identify non-zero weights
        non_zero_mask = flattened_weights != 0
        non_zero_weights = flattened_weights[non_zero_mask].cpu().numpy().reshape(-1, 1)

        # Apply K-means clustering to non-zero weights
        if len(non_zero_weights) > 0:
            min_weight = non_zero_weights.min()
            max_weight = non_zero_weights.max()
            init_centroids = np.linspace(min_weight, max_weight, n_clusters).reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=init_centroids)
            kmeans.fit(non_zero_weights)

            # Create the index matrix initialized with -1
            index_matrix = torch.full_like(flattened_weights, -1, device=weight_tensor.device, dtype=torch.long)

            # Update indices for non-zero weights
            index_matrix[non_zero_mask] = torch.tensor(kmeans.labels_, device=weight_tensor.device, dtype=torch.long)

           # Assigning the centroid values back to the non-zero weights
            centroids = torch.from_numpy(kmeans.cluster_centers_).to(weight_tensor.device)
            labels = torch.tensor(kmeans.labels_, device=weight_tensor.device)
            quantized_weights = centroids[labels]

            # Flatten the quantized weights to match the non-zero positions in the weight tensor
            quantized_weights_flattened = quantized_weights.view(-1)

            # Correctly assign the quantized weights to the non-zero positions
            # print('weight_tensor.data.view(-1).shape', weight_tensor.data.view(-1).shape)
            # print('non_zero_mask.view(-1).shape', non_zero_mask.view(-1).shape)
            # print('quantized_weights_flattened.shape', quantized_weights_flattened.shape)
            # input()
            weight_tensor.data.view(-1)[non_zero_mask.view(-1)] = quantized_weights_flattened

        else:
            # Handle the case where all weights are zero
            index_matrix = torch.full_like(flattened_weights, -1, device=weight_tensor.device, dtype=torch.long)

        return index_matrix.view(original_shape)


    def update_gradients(self):
        for name, module in self.model.named_modules():
            if name in self.index_matrices:
                self._aggregate_gradients(module, self.index_matrices[name])

    def _aggregate_gradients(self, module, index_matrix):
        if module.weight.grad is not None:
            # Flatten the gradient and index matrix
            grad = module.weight.grad.data.view(-1)
            indices = index_matrix.view(-1)

            # Create masks for indices
            valid_indices_mask = indices != -1
            invalid_indices_mask = indices == -1

            # Filter out gradients and indices where index is not -1
            valid_grads = grad[valid_indices_mask]
            valid_indices = indices[valid_indices_mask]

            # Create a tensor to hold the sum of gradients for each cluster
            grad_sum = torch.zeros_like(valid_grads).scatter_add_(0, valid_indices.long(), valid_grads)

            # Update gradients where index is not -1
            grad[valid_indices_mask] = grad_sum[valid_indices]

            # Zero out gradients where index is -1
            grad[invalid_indices_mask] = 0

            # Reshape the gradient tensor back to its original shape
            module.weight.grad.data = grad.view(module.weight.grad.data.shape)

