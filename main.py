from models import ModelLoader, ModelTrainer
from pruner import Pruner
from data import DatasetLoader
import torch
from utils import StatCollector, calculate_sparsity, summarize_model
from quantizer import Quantizer
import os
import json
from encoder import Encoder
from operator import itemgetter


class DeepCompression:
    def __init__(self, config):
        """
        Initializes the DeepCompression class with configuration settings.

        Args:
            config (dict): The configuration dictionary containing model parameters.
        """
        self.unpack_config(config)
        self.setup_environment(config)

    def unpack_config(self, config):
        keys = ('model_name', 'dataset_name', 'stage', 'learning_rate', 'epochs',
                'model_to_prune', 'pruning_learning_rate', 'pruning_threshold',
                'pruning_epochs', 'pruning_stages', 'model_to_quantize',
                'quantization_learning_rate', 'quantization_epochs',
                'conv_quantize_bit', 'fc_quantize_bit', 'model_to_encode', 'index_bit')

        for key in keys:
            setattr(self, key, config.get(key, None))  # Set attributes with values from config, or None if key is not present

    def setup_environment(self, config):
        """
        Sets up the environment including creating session directories, saving configurations,
        and initializing the model, data loaders, and statistics collector.
        """
        self.session_name = f"{self.model_name}_{self.dataset_name}"
        models_directory = f'models/{self.session_name}'
        os.makedirs(models_directory, exist_ok=True)

        config_path = os.path.join(models_directory, f"{self.stage}_config.json")
        with open(config_path, 'w') as config_file:
            # Assuming all config attributes are already set in self
            json.dump(config, config_file, indent=4)

        model_loader = ModelLoader()
        self.model = model_loader.load_model(self.model_name)

        dataset_loader = DatasetLoader(self.dataset_name)
        self.train_loader, self.test_loader = dataset_loader.load_data()

        self.stat_collector = StatCollector(folder=models_directory)

    def train_baseline(self):
        trainer = ModelTrainer(self.model, self.train_loader, self.test_loader, self.stat_collector, learning_rate=self.learning_rate, session_name=self.session_name)
        trainer.train_baseline(epochs=self.epochs)
        print('Finished Training Baseline')

    def perform_pruning(self):
        self.model.load_state_dict(torch.load(f'models/{self.session_name}/{self.model_to_prune}'), strict=False)
        trainer = ModelTrainer(self.model, self.train_loader, self.test_loader, self.stat_collector, learning_rate=self.pruning_learning_rate, session_name=self.session_name)
        trainer.train_and_prune(stages=self.pruning_stages, epochs=self.pruning_epochs, threshold=self.pruning_threshold)
        print('Finished Pruning Stage')

    def execute_quantization(self):
        self.model.load_state_dict(torch.load(f'models/{self.session_name}/{self.model_to_quantize}'), strict=True)
        total_sparsity, _ = calculate_sparsity(self.model)
        print(f'Sparsity of the model before quantization: {total_sparsity}')
        trainer = ModelTrainer(self.model, self.train_loader, self.test_loader, self.stat_collector, learning_rate=self.quantization_learning_rate, session_name=self.session_name)
        trainer.train_and_quantize(epochs=self.quantization_epochs, conv_bit=self.conv_quantize_bit, fc_bit=self.fc_quantize_bit)
        print('Finished Quantization Stage')

    def conduct_encoding(self):
        self.model.load_state_dict(torch.load(f'models/{self.session_name}/{self.model_to_encode}'), strict=True)
        total_sparsity, layer_sparsity = calculate_sparsity(self.model)
        encoder = Encoder()
        weight_summary = encoder.summarize_weights(self.model)
        position_summary, non_zero_lengths = encoder.encode_sparse_weights(self.model, self.index_bit)
        weight_huffman_encoding,  weight_average_code_lengths = encoder.huffman_encoding(weight_summary)
        position_huffman_encoding, position_average_code_lengths = encoder.huffman_encoding(position_summary)
        self.stat_collector.plot_distribution(position_summary, 'Position', 'before_encoding')
        self.stat_collector.plot_distribution(weight_summary, 'Weight', 'before_encoding')
        model_report = summarize_model(self.model, 
                        weight_huffman_encoding, 
                        weight_average_code_lengths,
                        position_huffman_encoding,
                        position_average_code_lengths,
                        self.index_bit,
                        non_zero_lengths, 
                        layer_sparsity, 
                        total_sparsity)
        # save model report as json
        with open(f'models/{self.session_name}/model_report.json', 'w') as f:
            json.dump(model_report, f, indent=4)
        print('Finished Encoding Stage')


    def main(self):
        # Stage logic
        if self.stage == 'baseline':
            self.train_baseline()
        elif self.stage == 'pruning':
            self.perform_pruning()
        elif self.stage == 'quantization':
            self.execute_quantization()
        elif self.stage == 'encoding':
            self.conduct_encoding()

if __name__ == '__main__':

    # based on the stage, the program will use different part of the config.
    config = {
        # model and dataset 
        'model_name': 'lenet5',
        'dataset_name': 'mnist',

        # stage
        'stage': 'quantization',

        # basline training
        'learning_rate': 1e-3,
        'epochs': 500,

        # pruning
        'model_to_prune': 'baseline_13.pth',
        'pruning_learning_rate': 1e-3,
        'pruning_epochs': 5, # number of epochs per pruning stage
        'pruning_stages': 50, # number of pruning stage
        'pruning_threshold': 0.12,

        # quantization
        'model_to_quantize': 'prune_24.pth',
        'quantization_learning_rate': 1e-4,
        'quantization_epochs': 100,
        'conv_quantize_bit': 8,
        'fc_quantize_bit': 5,

        # encoding and summarize
        'model_to_encode': 'quantized_20.pth',
        'index_bit': 5,
    }
    DeepCompression(config).main()