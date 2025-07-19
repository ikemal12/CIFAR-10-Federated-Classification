# CIFAR-10 Federated Classification

This project demonstrates a comprehensive federated learning pipeline for image classification on the CIFAR-10 dataset using the Flower framework. The implementation showcases the progression from centralized learning to advanced federated learning strategies, including custom aggregation methods and large-scale client simulation.

## Overview

Federated learning is a decentralized machine learning approach where models are trained locally on edge devices (clients) and then aggregated at a central server to form a global model. This approach preserves data privacy since raw data never leaves individual clients.

## Structure

The notebook follows a three-step progression:

### 1. **Centralized Learning Baseline**
- Traditional single-machine training with PyTorch
- Establishes performance benchmark on CIFAR-10
- 5-epoch training with validation tracking

### 2. **Standard Federated Learning**
- Implementation using Flower's `FedAvg` strategy
- Multiple simulation configurations:
  - Small-scale: 3 clients per round (10 total clients)
  - Full participation: All 10 clients per round
  - Weighted evaluation metrics aggregation

### 3. **Custom Strategy & Large-Scale Simulation**
- **Custom `FedCustom` Strategy**: Built from scratch implementing heterogeneous learning rates
  - Half the clients train with standard learning rate (0.001)
  - Half the clients train with higher learning rate (0.003)
- **Large-Scale Simulation**: 1000 clients with 1% participation per round (10/1000)
- Demonstrates scalability and real-world federated learning scenarios 

## Simulation Configurations

| Configuration | Clients | Participation | Rounds | Learning Rates |
|---------------|---------|---------------|--------|----------------|
| Baseline FL | 10 | 30% (3 clients) | 3 | 0.001 (uniform) |
| Full FL | 10 | 100% (10 clients) | 5 | 0.001 (uniform) |
| Custom Small | 10 | 30% (3 clients) | 2 | 0.001 & 0.003 (heterogeneous) |
| Custom Large | 1000 | 1% (10 clients) | 3 | 0.001 & 0.003 (heterogeneous) |

## Technologies Used

- **Flower 1.19.0**
- **PyTorch 2.7.1**
- **CIFAR-10**: Computer vision dataset (10 classes, 60K images)
