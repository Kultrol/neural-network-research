# Neural Network Pruning Analysis

## Background

Neural network pruning is a technique used to reduce the size and computational requirements of deep learning models by removing redundant or less important parameters, such as weights or entire neurons. Pruning can maintain, or sometimes even improve, model performance by encouraging sparsity and reducing overfitting. 

This program is designed to analyze the effects of pruning on various neural network architectures, ranging from simple models like Single Layer Perceptron (SLP) to more complex networks like Convolutional Neural Networks (CNN) and DenseNet-121. These architectures will be tested across several well-known datasets (MNIST, FMNIST, CIFAR-10) using multiple pruning techniques, including weight, neuron, structured, and unstructured pruning. The ultimate goal is to explore how different pruning configurations affect performance metrics like accuracy, training time, and computational efficiency.

## Requirements

The Neural Network Pruning Analysis system will have the following requirements:

### Must Have
- Support for three types of neural networks:
  - Single Layer Perceptron (SLP)
  - Convolutional Neural Networks (CNN)
  - DenseNet-121
- Integration with three datasets:
  - MNIST
  - FMNIST
  - CIFAR-10
- Ability to configure batch sizes (32, 256, 4000, and 60,000) for training across all models and datasets.
- Pruning techniques for different configurations:
  - Weight pruning
  - Neuron pruning
  - Structured pruning
  - Unstructured pruning
- Initial Performance Assessment (IPA) with metrics:
  - Accuracy
  - Cross-entropy loss
  - Training time (including FLOPs and Op-Counter)
- Functionality for testing different pruning percentages (0%-100%) in increments of 10%.
- Modular architecture that allows switching between models, datasets, pruning techniques, and configurations dynamically.

### Should Have
- Dynamic and adaptive pruning strategies that decide pruning during training.
- Op-Counter and FLOPs integration to track operations and compute efficiency.
- Re-run functionality for repeating experiments across different configurations and ensuring result reproducibility.

### Could Have
- Future expansion to support more complex datasets such as CIFAR-100.
- Transfer learning functionality, allowing for pruning on pre-trained models.
- Multi-GPU support to handle large-scale datasets and heavy models more efficiently.

## Method

### 1. System Architecture Overview

The architecture will follow a modular design, with a main controller managing different modules, including model loading, dataset handling, pruning application, and performance evaluation.

```plantuml
@startuml
actor User
participant "Main Controller" as MC
participant "Model Loader" as ML
participant "Dataset Loader" as DL
participant "Pruning Handler" as PH
participant "Training Manager" as TM
participant "Op-Counter & Metrics" as OC

User -> MC : Start Experiment
MC -> ML : Load Selected Model
MC -> DL : Load Selected Dataset
MC -> PH : Apply Pruning Configuration
MC -> TM : Train the Model
TM -> OC : Measure IPA and TL
MC -> User : Return Results (IPA, TL, Final Metrics)
@enduml
