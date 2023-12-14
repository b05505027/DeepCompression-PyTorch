import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn as nn
import torch
import json
from collections import defaultdict
import numpy as np

class StatCollector:
    def __init__(self, folder='default'):
        self.folder = folder
        self.iteration_loss = []
        self.epoch_accuracy = []
        self.stage_sparsity = []

    def log_loss(self, loss):
        self.iteration_loss.append(loss)

    def log_accuracy(self, accuracy):
        self.epoch_accuracy.append(accuracy)

    def log_sparsity(self, sparsity):
        self.stage_sparsity.append(sparsity)

    def clear_stats(self):
        self.iteration_loss = []
        self.epoch_accuracy = []
        self.stage_sparsity = []

    def plot_distribution(self, distribution, distribution_type, prefix):
        """
        Plots the distribution of weights or positions for each layer in a neural network.

        Args:
            distribution (dict): A dictionary containing layer names as keys and 
                                distribution values as items.
            distribution_type (str): The type of distribution (e.g., 'Weight', 'Position').
            prefix (str): Prefix for the saved plot file name.
        """
        num_layers = len(distribution)
        if num_layers > 0:
            plt.figure(figsize=(15, 4 * num_layers))
            cmap = plt.cm.viridis
            color_list = [mcolors.rgb2hex(cmap(i / num_layers)) for i in range(num_layers)]

            for i, (layer_name, values) in enumerate(distribution.items()):
                plt.subplot(num_layers, 1, i + 1)
                sorted_values = dict(sorted(values.items()))
                plt.bar(sorted_values.keys(), sorted_values.values(), color=color_list[i])

                plt.xlabel(f'{distribution_type} Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title(f'{distribution_type} Distribution in {layer_name}', fontsize=14)
                plt.xticks(rotation=45)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            plt.tight_layout()
            plt.savefig(f'{self.folder}/{prefix}_{distribution_type.lower()}_distribution.png')
            plt.close('all')

    
    def plot_stats(self, interval=50, prefix=""):
        """
        Plot statistics including loss per iteration, accuracy per epoch, and sparsity per stage
        in a colorful and professional style.
        
        Args:
            interval (int): Interval for plotting iteration loss.
            prefix (str): Prefix for the output file names.
        """
        plt.style.use('seaborn-darkgrid')  # Using seaborn style for better aesthetics

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))  # Adjust the size as needed

        # Check if there are stats to plot
        stats_available = any(len(stat) > 0 for stat in [self.iteration_loss, self.epoch_accuracy, self.stage_sparsity])
        
        if not stats_available:
            print("No statistics available to plot.")
            return

        # Plot loss per iteration
        if self.iteration_loss:
            axes[0].plot(range(0, len(self.iteration_loss), interval), self.iteration_loss[::interval], label='Loss per Iteration', linewidth=2, color='tab:blue')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss per Iteration')
            axes[0].legend()

        # Plot accuracy per epoch
        if self.epoch_accuracy:
            epochs = list(range(len(self.epoch_accuracy)))
            axes[1].plot(epochs, self.epoch_accuracy, label='Accuracy per Epoch', linewidth=2, color='tab:orange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy per Epoch')
            axes[1].legend()

        # Plot sparsity per stage
        if self.stage_sparsity:
            stages = list(range(len(self.stage_sparsity)))
            axes[2].plot(stages, self.stage_sparsity, label='Sparsity per Stage', linewidth=2, color='tab:green')
            axes[2].set_xlabel('Stage')
            axes[2].set_ylabel('Sparsity')
            axes[2].set_title('Sparsity per Stage')
            axes[2].legend()

        plt.tight_layout()
        plt.savefig(f'{self.folder}/{prefix}_stats.png')
        plt.close(fig)

        # Dump the stats to JSON
        stats_dict = {
            'iteration_loss': self.iteration_loss,
            'epoch_accuracy': list(zip(epochs, self.epoch_accuracy)),
            'stage_sparsity': list(zip(stages, self.stage_sparsity)),
        }
        with open(f'{self.folder}/{prefix}_stats.json', 'w') as f:
            json.dump(stats_dict, f, indent=4)

def calculate_sparsity(model):
    total_elements = 0
    zero_elements = 0
    layer_sparsity = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight'):
                layer_zeros = torch.sum(module.weight == 0).item()
                layer_elements = module.weight.numel()

                zero_elements += layer_zeros
                total_elements += layer_elements

                if layer_elements > 0:
                    layer_sparsity[name] = layer_zeros / layer_elements

    if total_elements == 0:
        return 0, {}  # To avoid division by zero

    total_sparsity = zero_elements / total_elements
    return total_sparsity, layer_sparsity


def summarize_model(model,
                    weight_huffman_encoding, 
                    weight_average_code_lengths,
                    position_huffman_encoding,
                    position_average_code_lengths,
                    index_bit,
                    non_zero_lengths, 
                    layer_sparsity, 
                    total_sparsity):
    
    summary = defaultdict(dict)

    # Assume model.named_parameters() provides the total number of parameters per layer
    params = {name: param.numel() for name, param in model.named_parameters()}
    # if the name belongs to a bias (*.bias), then we skip it
    total_params = {name: params[name] for name in params if '.bias' not in name}

    # Initialize total compressed sizes for both calculations
    total_compressed_size_pq = 0
    total_compressed_size_pqh = 0

    for layer, encoding in weight_huffman_encoding.items():
        # Retrieve total weights for each layer from the model's parameters
        total_weights = total_params[layer + ".weight"] if layer + ".weight" in total_params else 0

        # Calculate weights bits and index bits using log2 of the number of unique keys (huffman codes)
        weight_bits_pq = np.log2(len(encoding))
        index_bits_pq = np.log2(len(position_huffman_encoding[layer]))

        # Retrieve the average code lengths for weights and positions after Huffman encoding
        avg_weight_code_length_pqh = weight_average_code_lengths[layer]
        avg_position_code_length_pqh = position_average_code_lengths[layer]

        # Calculate compressed sizes
        compressed_size_pq = (index_bits_pq * (non_zero_lengths[layer] - 1)) + (weight_bits_pq * non_zero_lengths[layer]) + (np.power(2, weight_bits_pq) * 32)
        compressed_size_pqh = (avg_position_code_length_pqh * (non_zero_lengths[layer] - 1)) + (avg_weight_code_length_pqh * non_zero_lengths[layer] )+ (np.power(2, weight_bits_pq) * 32)
        
        # Add to the total compressed sizes
        total_compressed_size_pq += compressed_size_pq
        total_compressed_size_pqh += compressed_size_pqh
        
        # Calculate compression rates for the layer
        compression_rate_pq = 100 * (compressed_size_pq / (32 * total_weights))
        compression_rate_pqh = 100 * (compressed_size_pqh / (32 * total_weights))
        
        # Store the information in the summary dictionary
        summary[layer]['#Weights'] = total_weights
        summary[layer]['Weights%'] = (1 - layer_sparsity[layer]) * 100
        summary[layer]['Weight bits (P+Q)'] = weight_bits_pq
        summary[layer]['Weight bits (P+Q+H)'] = avg_weight_code_length_pqh
        summary[layer]['Index bits (P+Q)'] = index_bits_pq
        summary[layer]['Index bits (P+Q+H)'] = avg_position_code_length_pqh
        summary[layer]['Compress rate (P+Q)'] = f"{compression_rate_pq:.2f}%"
        summary[layer]['Compress rate (P+Q+H)'] = f"{compression_rate_pqh:.2f}%"

    # Calculate the original size of the model
    total_original_size = sum(total_params.values()) * 32

    # Calculate the total compression rates
    total_compression_rate_pq = 100 * (total_compressed_size_pq / total_original_size)
    total_compression_rate_pqh = 100 * (total_compressed_size_pqh / total_original_size)

    summary['Total']['#Weights'] = sum(total_params.values())
    summary['Total']['Weights%'] = (1 - total_sparsity) * 100
    summary['Total']['Compress rate (P+Q)'] = f"{total_compression_rate_pq:.2f}%"
    summary['Total']['Compress rate (P+Q+H)'] = f"{total_compression_rate_pqh:.2f}%"

    return dict(summary)  # Convert from defaultdict to dict for the final summary