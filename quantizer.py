import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

class Quantizer:
    def __init__(self, model):
        self.model = model
        self.index_matrices = {}  # To store index matrix for each module
    
    def load_index_matrices(self, folder='default'):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                self.index_matrices[name] = torch.load(f'{folder}/index_matrices/' + name + '.pth')

    def quantize_weights(self, folder='default'):
        for name, module in tqdm(self.model.named_modules(), desc='Quantizing weights', total=len(list(self.model.named_modules()))):
            if isinstance(module, torch.nn.Conv2d):
                self.index_matrices[name] = self._create_index_matrix(module.weight, n_clusters=256)
            elif isinstance(module, torch.nn.Linear):
                self.index_matrices[name] = self._create_index_matrix(module.weight, n_clusters=32)
            else:
                continue
            # save index matrix, which contains tensor of indices
            os.makedirs(f'{folder}/index_matrices', exist_ok=True)
            torch.save(self.index_matrices[name], f'{folder}/index_matrices/' + name + '.pth')




    def _create_index_matrix(self, weight_tensor, n_clusters):
        # Flatten the weight tensor for K-means
        original_shape = weight_tensor.shape
        flattened_weights = weight_tensor.detach().view(-1, 1).cpu().numpy()

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(flattened_weights)

        # Create and return the index matrix
        index_matrix = torch.tensor(kmeans.labels_, device=weight_tensor.device).view(original_shape)

        # Assigning the centroid values back to the weights
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(weight_tensor.device)
        labels = torch.from_numpy(kmeans.labels_).to(weight_tensor.device)
        quantized_weights = centroids[labels].view(original_shape)
        weight_tensor.data = quantized_weights
        return index_matrix

    def update_gradients(self):
        for name, module in self.model.named_modules():
            if name in self.index_matrices:
                self._aggregate_gradients(module, self.index_matrices[name])

    def _aggregate_gradients(self, module, index_matrix):
        if module.weight.grad is not None:
            # Reshape the gradient to match the weights
            grad = module.weight.grad.data
            grad_shape = grad.shape
            grad = grad.view(-1, 1)

            # Aggregate gradients within each cluster
            for i in range(torch.max(index_matrix) + 1):
                mask = (index_matrix.view(-1, 1) == i)
                if torch.sum(mask) > 0:
                    mean_grad = torch.mean(grad[mask])
                    grad[mask] = mean_grad

            # Reshape the gradients back to their original shape
            module.weight.grad.data = grad.view(grad_shape)
