import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import json
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
    
    def plot_stats(self, interval=10, prefix=""):
        # Plot loss per iteration
        if len(self.iteration_loss) > 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(self.iteration_loss[::interval], label='Loss per Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss per Iteration')
            plt.legend()

        # Plot accuracy per epoch
        if len(self.epoch_accuracy) > 0:
            plt.subplot(1, 3, 2)
            plt.plot(self.epoch_accuracy, label='Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy per Epoch')
            plt.legend()

        # Plot sparsity per stage
        if len(self.stage_sparsity) > 0:
            plt.subplot(1, 3, 3)
            plt.plot(self.stage_sparsity, label='Sparsity per Stage')
            plt.xlabel('Stage')
            plt.ylabel('Sparsity')
            plt.title('Sparsity per Stage')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.folder}/{prefix}_stats.png')
        # also dump the stats to json
        with open(f'{self.folder}/{prefix}_stats.json', 'w') as f:
            json.dump({'iteration_loss': self.iteration_loss, 'epoch_accuracy': self.epoch_accuracy, 'stage_sparsity': self.stage_sparsity}, f)
        plt.close('all')

def calculate_sparsity(model):
    total_elements = 0
    zero_elements = 0

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight'):
                zero_elements += torch.sum(module.weight == 0).item()
                total_elements += module.weight.numel()

    if total_elements == 0:
        return 0  # To avoid division by zero

    sparsity = zero_elements / total_elements
    return sparsity
