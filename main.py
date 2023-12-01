from models import ModelLoader, ModelTrainer
from pruner import Pruner
from data import DatasetLoader
import torch
from utils import StatCollector, calculate_sparsity
from quantizer import Quantizer
import os
import json

def main(config):
    # Load config
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    stage = config['stage']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    pruning_learning_rate = config['pruning_learning_rate']
    pruning_threshold = config['pruning_threshold']
    pruning_epochs = config['pruning_epochs']
    pruning_stages = config['pruning_stages']
    quantization_learning_rate = config['quantization_learning_rate']
    conv_quantize_bit = config['conv_quantize_bit']
    fc_quantize_bit = config['fc_quantize_bit']
    index_bit = config['index_bit']



    # Create a session name
    session_name = f'{model_name}_{dataset_name}'
    os.makedirs(f'models/{session_name}', exist_ok=True)

    # Save config
    with open(f'models/{session_name}/{stage}_config.json', 'w') as f:
        json.dump(config, f)

    # Initialize model
    loader = ModelLoader()
    model = loader.load_model(model_name)

    # Define data loading
    dataset_loader = DatasetLoader(dataset_name)
    trainloader, testloader = dataset_loader.load_data()

    # Initialize stat collector
    stat_collector = StatCollector(folder=f'models/{session_name}')


    if stage == 'baseline':
        # Baseline Training
        trainer = ModelTrainer(model, trainloader, testloader, stat_collector, learning_rate=learning_rate, session_name=session_name)
        trainer.train_baseline(epochs=epochs)
        print('Finish Training Baseline')

    elif stage == 'pruning':
        # Pruning Stage
        trainer = ModelTrainer(model, trainloader, testloader, stat_collector, learning_rate=pruning_learning_rate, session_name=session_name)
        model.load_state_dict(torch.load(f'models/{session_name}/baseline_26.pth'), strict=False)
        trainer.train_and_prune(stages=pruning_stages, epochs=pruning_epochs, threshold=pruning_threshold)
        print('Finish Pruning Stage')
    elif stage == 'quantization':
        # Quantization Stage
        model.load_state_dict(torch.load('model_prune_stage_3.pth'), strict=False)


if __name__ == '__main__':
    config = {
        # model and dataset 
        'model_name': 'lenet5',
        'dataset_name': 'mnist',
        # stage
        'stage': 'baseline',
        # basline training
        'learning_rate': 1e-2,
        'epochs': 30,
        # pruning
        'pruning_learning_rate': 1e-5,
        'pruning_epochs': 10, # number of epochs per pruning stage
        'pruning_stages': 50, # number of pruning stage
        'pruning_threshold': 0.0425,
        # quantization
        'quantization_learning_rate': 1e-5,
        'conv_quantize_bit': 8,
        'fc_quantize_bit': 6,
        # encoding
        'index_bit': 5
    }
    # for i in range(10):
    #     config['learning_rate'] = 1e-3 + i * 2e-4
    main(config)