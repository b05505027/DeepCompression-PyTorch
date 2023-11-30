from models import ModelLoader, ModelTrainer
from pruner import Pruner
from data import DatasetLoader
import torch
from utils import StatCollector, calculate_sparsity
from quantizer import Quantizer
import os

# Example usage
model_name = 'alexnet'
dataset_name = 'cifar10'
session_name=f'{model_name}_{dataset_name}'
loader = ModelLoader()
model = loader.load_model(model_name)
os.makedirs(f'models/{session_name}', exist_ok=True)

# Define data loading
dataset_loader = DatasetLoader(dataset_name)
trainloader, testloader = dataset_loader.load_data()

# Initialize stat collector
stat_collector = StatCollector(folder=f'models/{session_name}')

# Initialize model trainer
trainer = ModelTrainer(model, trainloader, testloader, stat_collector, learning_rate=5e-4, session_name=session_name )

# Initial Training
trainer.train_baseline(epochs=10)
print('Finish Training Baseline')

# Pruning Stage
trainer.train_and_prune(stages=10, epochs=1)
print('Finish Pruning Stage')

# Quantization Stage
# model.load_state_dict(torch.load('model_prune_stage_3.pth'), strict=False)
trainer.train_and_quantize(epochs=10)
print('Finish Quantization Stage')
