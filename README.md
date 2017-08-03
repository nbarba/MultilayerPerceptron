# MultilayerPerceptron
Theano wrapper for creating a simple multi-layer perceptron neural network.

# Usage

The use of the MLP class is pretty straightforward. The neural network must first be instantiated  and the method to train the network must be invoked, i.e.: 

```
classifier = MLP(number_of_features,number_of_neurons_in_hidden_layer)
classifier.minibatch_gradient_descent(features,labels,batch_size);

```
Constructor parameters include the number of neurons for the input and hidden layer. The minibatch gradient descent algorithm is implemented, so the batch size must also be passed as an argument. Other training parameters that can be defined, include
  - the learning rate
  - the l2 normalization lambda value

# Toy Example

A toy example is also provided, that attempts to train the multi-layer perceptron on a real binary classification dataset. The dataset used is a randomly selected subset of the "banknote authentication" dataset available at the [UCI Machine learning repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). The task is to distinguish original from forged bank-note like statements, using wavelet features extracted from images. The dataset is un-normalized, so the example also includes code to perform z-score normalization on the train/test set. 



