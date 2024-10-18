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


## 2. Model Selection

The program will support three types of models:
  - **Single Layer Perceptron (SLP)**: A basic neural network architecture.
  - **Convolutional Neural Network (CNN)**: A more complex architecture with layers like convolution, pooling, and fully connected layers.
  - **DenseNet-121**: A state-of-the-art network known for its dense connections and feature reuse across layers.

```python
class ModelLoader:
    def load_model(self, model_type):
        if model_type == "SLP":
            return self.load_slp_model()
        elif model_type == "CNN":
            return self.load_cnn_model()
        elif model_type == "DenseNet-121":
            return self.load_densenet_model()
        else:
            raise ValueError("Invalid model type")

## 3. Dataset Handling

The system will integrate with three datasets—MNIST, FMNIST, and CIFAR-10. The dataset loader will dynamically load and preprocess the selected dataset and configure batch sizes for training.

```python
class DatasetLoader:
    def load_dataset(self, dataset_name, batch_size):
        if dataset_name == "MNIST":
            return self.load_mnist(batch_size)
        elif dataset_name == "FMNIST":
            return self.load_fmnist(batch_size)
        elif dataset_name == "CIFAR-10":
            return self.load_cifar10(batch_size)
        else:
            raise ValueError("Invalid dataset")


## 4. Pruning Strategies

The system will implement multiple pruning techniques, with pruning percentages applied in increments of 10%, from 0% to 100%, for each batch size (32, 256, 4000, and 60,000):

- **Weight Pruning**: Removes individual weights based on L1/L2 norm or a predefined threshold.
- **Neuron Pruning**: Removes entire neurons (or channels in CNNs) based on their contribution to the network's output.
- **Structured Pruning**: Removes groups of parameters, such as entire filters or layers, while preserving the overall structure.
- **Unstructured Pruning**: Prunes individual weights without maintaining structural consistency.

```python
class PruningHandler:
    def apply_pruning(self, model, pruning_type, percentage):
        if pruning_type == "weight":
            self.weight_pruning(model, percentage)
        elif pruning_type == "neuron":
            self.neuron_pruning(model, percentage)
        elif pruning_type == "structured":
            self.structured_pruning(model, percentage)
        elif pruning_type == "unstructured":
            self.unstructured_pruning(model, percentage)
        else:
            raise ValueError("Invalid pruning type")


## 5. Performance Metrics and Evaluation

To measure performance and efficiency, the system will calculate:

- **Initial Performance Assessment (IPA)**: Includes accuracy, cross-entropy loss, and training time.
- **Training Time (TL)**: Measured using the Op-Counter and FLOPs to track the number of operations during forward and backpropagation.
- **Final Metrics**: Evaluated after pruning, including final accuracy and cross-entropy.

```python
class OpCounter:
    def calculate_operations(self, model, data_loader):
        # Count operations per forward and backward pass
        pass

    def calculate_flops(self, model, data_loader):
        # Calculate floating-point operations (FLOPs)
        pass


## 6. Re-runs and Replication

The program will support the re-running of experiments across different configurations (pruning percentage, model type, dataset) to ensure result reproducibility. Results can be exported for further analysis or comparison.

```python
class ExperimentController:
    def run_experiment(self, model_type, dataset_name, batch_size, pruning_type, pruning_percentage):
        model = self.model_loader.load_model(model_type)
        dataset = self.dataset_loader.load_dataset(dataset_name, batch_size)
        self.pruning_handler.apply_pruning(model, pruning_type, pruning_percentage)
        metrics = self.training_manager.train(model, dataset)
        return metrics
