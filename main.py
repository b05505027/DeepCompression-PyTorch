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

    model_to_prune = config['model_to_prune']
    pruning_learning_rate = config['pruning_learning_rate']
    pruning_threshold = config['pruning_threshold']
    pruning_epochs = config['pruning_epochs']
    pruning_stages = config['pruning_stages']

    model_to_quantize = config['model_to_quantize']
    quantization_learning_rate = config['quantization_learning_rate']
    quantization_epochs = config['quantization_epochs']
    conv_quantize_bit = config['conv_quantize_bit']
    fc_quantize_bit = config['fc_quantize_bit']
    index_bit = config['index_bit']



    # Create a session name
    session_name = f'{model_name}_{dataset_name}'
    os.makedirs(f'models/{session_name}', exist_ok=True)

    # Save config
    with open(f'models/{session_name}/{stage}_config.json', 'w') as f:
        # save config with indent 4
        json.dump(config, f, indent=4)

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
        model.load_state_dict(torch.load(f'models/{session_name}/{model_to_prune}'), strict=False)
        trainer.train_and_prune(stages=pruning_stages, epochs=pruning_epochs, threshold=pruning_threshold)
        print('Finish Pruning Stage')

    elif stage == 'quantization':
        # Quantization Stage
        sparsity = calculate_sparsity(model)
        print('Sparsity of the model before quantization:', sparsity)

        print('model_to_quantize', model_to_quantize)
        model.load_state_dict(torch.load(f'models/{session_name}/{model_to_quantize}'), strict=True)
        print('Finish Loading Model')


        sparsity = calculate_sparsity(model)
        print('Sparsity of the model before quantization:', sparsity)
        input()

        trainer = ModelTrainer(model, trainloader, testloader, stat_collector, learning_rate=quantization_learning_rate, session_name=session_name)
        trainer.train_and_quantize(epochs=quantization_epochs, conv_bit=conv_quantize_bit, fc_bit=fc_quantize_bit)
    
    elif stage == 'encoding':
        # Encoding Stage
        # sparsity = calculate_sparsity(model)
        # print('Sparsity of the model before encoding:', sparsity)
        model.load_state_dict(torch.load(f'models/{session_name}/{model_to_encode}'), strict=True)

        
if __name__ == '__main__':
    config = {
        # model and dataset 
        'model_name': 'lenet300',
        'dataset_name': 'mnist',
        # stage
        'stage': 'encoding',
        # basline training
        'learning_rate': 3e-3,
        'epochs': 30,
        # pruning
        'model_to_prune': 'baseline_26.pth',
        'pruning_learning_rate': 1e-4,
        'pruning_epochs': 5, # number of epochs per pruning stage
        'pruning_stages': 50, # number of pruning stage
        'pruning_threshold': 0.068,

        # quantization
        'model_to_quantize': 'prune_50.pth',
        'quantization_learning_rate': 1e-4,
        'quantization_epochs': 100,
        'conv_quantize_bit': 8,
        'fc_quantize_bit': 6,

        # encoding
        'model_to_encode': 'quantize_100.pth',
    }
    main(config)