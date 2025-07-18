# CIFAR-10 Federated Classification

This project explores image classification on the CIFAR-10 dataset using federated learning with the Flower framework. Federated learning is a decentralized machine learning approach where models are 
trained locally on edge devices (known as clients) and then sent to a server which aggregates the updated weights to form a global model. This approach preserves data privacy since data never leaves the 
individual clients.

Specifically I have simulated federated learning with 10 clients and used FedAvg for aggregation. 

This is a good baseline experiment, but there are ways to improve it:

* Make my own custom aggregation strategy for better client selection
* Extend to larger scale of clients to better simulate real-world scenarios
