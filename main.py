from models import ModelLoader, ModelTrainer
from pruner import Pruner
from data import DatasetLoader
import torch
from utils import StatCollector, calculate_sparsity

# Example usage
loader = ModelLoader()
model = loader.load_model('alexnet')

# Define data loading
dataset_loader = DatasetLoader('cifar10')
trainloader, testloader = dataset_loader.load_data()

# Initialize stat collector
stat_collector = StatCollector()

# Initailize trainer and pruner
trainer = ModelTrainer(model, trainloader, testloader, stat_collector, learning_rate=5e-4)
pruner = Pruner(pruning_threshold=1e-4)


num_stages = 10
num_epochs = 3
sparsity = calculate_sparsity(model)
print(f"Sparsity before pruning: {sparsity:.6%}")
for stage in range(num_stages):
    print(f"Stage {stage + 1}/{num_stages}")

    # Train the model
    trainer.train_and_evaluate(num_epochs)

    # Prune the network
    model = pruner.prune_network(model)

    # Calculate and print sparsity
    sparsity = ModelTrainer.calculate_sparsity(model)
    print(f"Sparsity after stage {stage + 1}: {sparsity:.6%}")

    stat_collector.log_sparsity(sparsity)

    # Plotting the statistics after each stage
    stat_collector.plot_stats(interval=10)