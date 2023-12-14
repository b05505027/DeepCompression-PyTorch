
# PyTorch Implementation of Deep Compression

## Overview
This project presents a PyTorch-based implementation of the  Deep Compression algorithm, designed for efficient compression of deep learning models. The focus is on minimizing the size of neural networks without compromising their performance. Key techniques employed include pruning, quantization, and Huffman Encoding.

## Modules
- **main.py**: Entry point of the project with the `DeepCompression` class.
- **models.py**: Contains classes for model loading and training.
- **utils.py**: Offers statistical functions and the `StatCollector` class for data collection.
- **pruner.py**: Includes the `Pruner` class for network pruning.
- **quantizer.py**: Contains the `Quantizer` class for quantizing weights.
- **encoder.py**: Houses the `Encoder` class for Huffman Encoding.
- **lenets.py**: Provides LeNet model classes (Lenet300 and LeNet5).
- **data.py**: Offers dataset loaders and transformers.


## Usage
Configure the `config` dictionary in `main.py` based on the desired stage: baseline training, pruning, quantization, or encoding. 

Example:
```python
config = {
    'model_name': 'lenet300',
    'dataset_name': 'mnist',
    'stage': 'quantization',
    ...
}
DeepCompression(config).main()
```

## Configurations
- `model_name`: Name of the model (e.g., 'lenet300', 'lenet5').
- `dataset_name`: Name of the dataset (e.g., 'mnist', 'cifar10').
- `stage`: Stage to perform ('baseline', 'pruning', 'quantization', 'encoding').
- `learning_rate`, `epochs`: Parameters for baseline training.
- Pruning parameters: `model_to_prune`, `pruning_learning_rate`, etc.
- Quantization parameters: `model_to_quantize`, `quantization_learning_rate`, etc.
- Encoding parameters: `model_to_encode`, `index_bit`.

