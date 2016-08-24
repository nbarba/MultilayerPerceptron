""" Implementation of a multilayer perceptron using Theano  """

import theano
import theano.tensor as T
import numpy as np
from random import random 
import theano.tensor.nnet
from theano.compile.debugmode import DebugMode
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing

import matplotlib.pyplot as plt

class MLP(object):

    def __init__(self,feature_size,hidden_layer_size=10):
        
        #training parameters
        self._learning_rate = 0.04
        self._l2_lambda = 0.001
        self._max_epochs=200
        self._convergence_error=0.01
        self.hidden_layer_size=hidden_layer_size;


        #initialize the input variables (vectors because we'll have stochastic gradient descent, so we need one example at a time)
        input_values=T.matrix('input_value')      
        target_values=T.matrix('target_values')

        #define and initialize the weights
        weights_hidden=T.theano.shared(np.random.rand(feature_size,hidden_layer_size),'weights_hidden');
        weights_output=T.theano.shared(np.random.rand(hidden_layer_size,1),'weights_output')
        bias_hidden=T.theano.shared(np.ones(hidden_layer_size),'bias_hidden');
        bias_output=T.theano.shared(np.ones(1),'bias_output');

        #define formulas to compute the output of each layer
        output_hidden=T.nnet.sigmoid(T.dot(input_values,weights_hidden)+bias_hidden);
        predicted_values=T.nnet.sigmoid(T.dot(output_hidden,weights_output)+bias_output);

        #define cost function & perform L2 regularisation
        cost = T.sqr(predicted_values - target_values).sum() 
        cost += 0.5*self._l2_lambda * (theano.tensor.sqr(weights_hidden).sum() + theano.tensor.sqr(weights_output).sum())

        #define gradients and update rules
        gradients=T.grad(cost,wrt=[weights_hidden,weights_output,bias_hidden,bias_output])
        updates = [(weight, weight - (self._learning_rate * deltas)) for weight, deltas in zip([weights_hidden, weights_output,bias_hidden,bias_output], gradients)]


        #define theano train - test functions
        self.train=theano.function(
            inputs=[input_values,target_values],
            outputs=[predicted_values,cost],
            updates=updates
        )
        self.test=theano.function(
            inputs=[input_values,target_values],
            outputs=[predicted_values,cost]        
        )


    def minibatch_gradient_descent(self,features_train,labels_train,batch_size):
        """ high level method for minibatch gradient descent """

        assert len(features_train) == len(labels_train)
        total_batches,remaining_instances=divmod(len(features_train),batch_size);
        epoch=1;
        cost_sum=self._convergence_error+1; #initialize cost_sum above convergence_error

        while cost_sum > self._convergence_error and epoch < self._max_epochs:        
            #shuffle training set
            features_shuffled,labels_shuffled=shuffle_inplace(features_train,labels_train)

            cost_sum = 0.0
            predicted_labels=[]

            #train network each batch
            for x in range(0,total_batches):
                batch_start=x*batch_size; 
                temp_features=features_shuffled[batch_start:batch_start+batch_size];
                temp_labels=labels_shuffled[batch_start:batch_start+batch_size];
                pred,cost=self.train(temp_features,(np.array(temp_labels)).reshape(((len(temp_labels)),1)))
                cost_sum+=cost
                predicted_labels.extend(pred)
            

            #train networkk for the remaining instances (if any)
            if (remaining_instances>0):
                temp_features=features_shuffled[batch_start:batch_start+remaining_instances];
                temp_labels=labels_shuffled[batch_start:batch_start+remaining_instances];
                pred,cost=self.train(temp_features,(np.array(temp_labels)).reshape(((len(temp_labels)),1)))
                cost_sum+=cost
                predicted_labels.extend(pred)
     
            # compute accuracy
            acc=accuracy_score(labels_shuffled, threshold_labels(predicted_labels,0.5))
            
            print "Epoch " + str(epoch)+": Training Cost="+str(cost_sum)+", Training Accuracy="+str(acc)
            epoch+=1;


    @property
    def max_epochs(self):
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, value):
        self._max_epochs = value

    @property
    def convergence_error(self):
        return self._convergence_error

    @convergence_error.setter
    def convergence_error(self, value):
        self._convergence_error = value


def shuffle_inplace(array1, array2):
    """ Shuffle two arrays but keep correspondence between elements """

    assert len(array1) == len(array2)
    p = np.random.permutation(len(array1))

    return np.array(array1)[p], np.array(array2)[p]


def load_dataset(filename):
    """ load dataset from file """

    dataset = []
    with open(filename, "r") as f:
        for line in f:
            split = line.strip().split(',')
            label = float(split[4])
            features = np.array([float(split[i]) for i in xrange(0, len(split)-1)])
            dataset.append((label, features))
    return dataset


def threshold_labels(labels,threshold_value):
    """ utility method used when computing the accuracy """
    tmp=np.array(labels)
    tmp[tmp>threshold_value]=1
    tmp[tmp<=threshold_value]=0

    return tmp

if __name__ == "__main__":
    
    train_filename="./data/data_banknote_authentication_train.txt"
    test_filename="./data/data_banknote_authentication_test.txt";

    #load training set
    train_set=load_dataset(train_filename);
    features_train=[train_set[i][1] for i in range(0,len(train_set))];
    labels_train=[train_set[i][0] for i in range(0,len(train_set))];

    #normalize training set 
    training_mean=np.mean(features_train)
    training_std=np.std(features_train)
    features_train_scaled= (features_train - training_mean) / training_std
  
    # train network
    print "----- Instantiating / training network ------"
    number_of_features=len(features_train_scaled[1])
    classifier=MLP(number_of_features)
    classifier.minibatch_gradient_descent(features_train_scaled,labels_train,10);
    print "\n----- Training done! ------"


    # test network
    print "\n----- Evaluating performance on test set ------"
    test_set=load_dataset(test_filename);
    features_test=[test_set[i][1] for i in range(0,len(test_set))];    
    labels_test=[test_set[i][0] for i in range(0,len(test_set))]; 

    features_test_scaled= (features_test - training_mean) / training_std
    predicted_labels,cost_iter=classifier.test(features_test_scaled,np.array(labels_test).reshape(((len(test_set)),1)))

    print(classification_report(labels_test,threshold_labels(predicted_labels,0.5)))


