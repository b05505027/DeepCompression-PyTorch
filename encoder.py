import torch
import torch.nn as nn
from collections import Counter
import numpy as np
from heapq import heappush, heappop, heapify
from collections import defaultdict


class Encoder:
    def __init__(self):
        self.weight_summary = {}
        self.position_summary = {}
        self.huffman_codes = {}

    def encode_sparse_weights(self, model, index_bit):

        position_summary = {}
        non_zero_lengths = {}
        max_index_diff = np.power(2, index_bit)
        for name, layer in model.named_modules():
            # Check if it's a convolutional layer or a linear layer
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Extract the weights
                flat_weights = next(layer.parameters()).data.flatten()

           
                non_zero_indices = torch.nonzero(flat_weights, as_tuple=False).view(-1)
                non_zero_values = flat_weights[non_zero_indices]

                
                # Initialize the storage for values and index differences
                values = []
                index_diffs = []
                
                # The first index doesn't need a difference calculation
                prev_index = non_zero_indices[0].item()
                values.append(non_zero_values[0].item())


                
                for idx in non_zero_indices[1:]:
                    value = flat_weights[idx].item()
                    diff = idx.item() - prev_index # Calculate difference from the previous index

                    
                    # Check if the difference exceeds the maximum index difference
                    while diff > max_index_diff:
                        # Insert filler zeros until the difference is within the allowed range
                        values.append(0.0)
                        index_diffs.append(max_index_diff)
                        diff -= max_index_diff
                    
                    values.append(value)
                    index_diffs.append(diff)
                    prev_index = idx.item()

                # length of values
                non_zero_length = len(values)
                position_count = {}
                for diff in index_diffs:
                    if diff in position_count:
                        position_count[diff] += 1
                    else:
                        position_count[diff] = 1

                position_summary[name] = position_count
                non_zero_lengths[name] = non_zero_length

        return position_summary, non_zero_lengths

    def summarize_weights(self, model):
        weight_summary = {}
        for name, layer in model.named_modules():
            # Check if it's a convolutional layer or a linear layer
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Extract the weights
                weights = next(layer.parameters()).data.numpy().flatten()

                # Filter out zero values
                non_zero_weights = weights[weights != 0]

                # Round the non-zero weights
                rounded_weights = np.round(non_zero_weights, decimals=6)
                # Count occurrences of each weight value (excluding zeros)
                weight_counts = Counter(rounded_weights)
                # Store the top 64 most common weights (excluding zeros)
                weight_summary[name] = dict(weight_counts)
                # Make their keys strings
                weight_summary[name] = {str(k): v for k, v in weight_summary[name].items()}

        self.weight_summary = weight_summary
        return weight_summary

    
    def get_average_code_length(self, token_frequencies, huffman_codes):
        total_symbols = sum(token_frequencies.values())
        total_weighted_code_length = sum(len(code) * token_frequencies[symbol] for symbol, code in huffman_codes.items())

        return total_weighted_code_length / total_symbols if total_symbols > 0 else 0


    def huffman_encoding(self, weight_summary):

        huffman_codes = {}
        average_code_lengths = {}
        
        # The wegh_summary is a dictionary of dictionaries
        # The outer dictionary has keys corresponding to layer names
        # The inner dictionaries have keys corresponding to weights
        # For each layer, we need to create a min heap of weights according to their frequencies
        for layer_name, weights in weight_summary.items():
            # Create a min heap of weights according to their frequencies
            heap = [[freq, [weight]] for weight, freq in weights.items()]
            heapify(heap) 
            codebook = defaultdict(str) # initialize the codebook for this layer



            # Merge the two smallest elements until only one is left
            while len(heap) > 1:
                lo = heappop(heap) # Pop the smallest item, lo stands for low.
                hi = heappop(heap) # Pop the next smallest item, hi stands for high.


                for weight in lo[1]:
                    codebook[weight] = '0' + codebook[weight]
                for weight in hi[1]:
                    codebook[weight] = '1' + codebook[weight]

                newnode_freq = lo[0] + hi[0]
                newnode_name = lo[1] + hi[1]
                newnode = [newnode_freq, newnode_name]

                heappush(heap, newnode)
            
            huffman_codes[layer_name] = codebook
            average_code_lengths[layer_name] = self.get_average_code_length(weights, codebook)
        
        return huffman_codes, average_code_lengths
