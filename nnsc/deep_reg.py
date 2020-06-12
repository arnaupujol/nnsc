"""
DEEP REG

This module defines the methods to make a regression from a deep neural network.

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0.0
"""


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()#It enables to use the behaviour from tensorflow 1
import scipy.io as sio
from copy import deepcopy as dp

def leaky_relu(z, name=None):
  """
  This method operates a Leaky ReLU on the value z.

  Parameters:
  -----------
  z: float
    Value to which we operate.
  name: str
    A name for the operation (optional).

  Returns:
  --------
  float
    Result of Leaky ReLU operation.
  """
  return tf.maximum(0.01 * z, z, name=name)

def np_leaky_relu(z):
  """
  This method is a numpy version of Leaky ReLU.

  Parameters:
  -----------
  z: float
    Value to which we operate.

  Returns:
  --------
  float
    Result of Leaky ReLU operation.
  """
  return np.maximum(0.01 * z, z)

def deep_reg(dimensions=[784, 512, 256, 64],ct=[0.1,0.1,0.1,0.1], activation = 'leaky_relu'):
    """
    This method builds the architecture corresponding to a NN regression.

    Parameters:
    -----------
    dimensions: list of ints
        List of number of neurons per layer.
    ct: list of floats
        List of values of the contamination level applied to the NN per layer.
    activation: str {'leaky_relu', 'relu_tanh', 'leaky_relu_l', 'tanh'}
        It defines the activation functions applied (default is 'leaky_relu')

            'leaky_relu':
                Leaky ReLU for all activation functions

            'relu_tanh':
                tanh on last hidden layer, Leaky ReLU for the rest

            'leaky_relu_l':
                Linear function on last hidden layer, Leaky ReLU for the rest

            'tanh':
                tanh for all activation functions

    Returns:
    --------
    dict
        Dictionary with:

            x: np.ndarray
                Input training data

            theta: np.ndarray
                Labels for supervised learning

            z: np.ndarray
                NN output

            W: np.ndarray
                Weights of NN

            b: np.ndarray
                Bias

            cost: float
                Value of cost function
    """
    L = len(dimensions) -1

    # INPUT DATA

    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    theta = tf.placeholder(tf.float32, [None, dimensions[-1]], name='theta')
    pcost = tf.placeholder(tf.float32, [1], name='pcost')

    # NOISY ENCODER

    encoder = []
    all_h = []
    b_enc = []
    noise = tf.random_normal(shape=tf.shape(x),stddev=ct[0],dtype=tf.float32)
    current_input = x + noise
    all_h.append(current_input)

    for layer_i in range(1,L+1):

        " Defining the variables "

        n_input = dimensions[layer_i-1]
        n_output = dimensions[layer_i]

        low = -np.sqrt(6.0/(n_input + n_output))
        high = np.sqrt(6.0/(n_input + n_output))
        nameW = 'Weights'
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              minval=low,
                              maxval=high),name=nameW.format(layer_i - 1))
        nameB = 'Bias'
        be = tf.Variable(tf.zeros([n_output]),name=nameB.format(layer_i - 1))

        b_enc.append(be)
        encoder.append(W)

        if layer_i == L+1:
            output = tf.matmul(current_input, W) + be
        else:
            if activation == 'leaky_relu' or (activation in ['relu_tanh', 'leaky_relu_l'] and layer_i < L):
                output = leaky_relu(tf.matmul(current_input, W) + be)
            elif activation == 'leaky_relu_l' and layer_i == L:
                output = tf.matmul(current_input, W) + be
            elif activation == 'tanh' or (activation == 'relu_tanh' and layer_i == L):
                output = tf.tanh(tf.matmul(current_input, W) + be)
        noise = tf.random_normal(shape=tf.shape(output),stddev=ct[layer_i],dtype=tf.float32)
        current_input = output + noise

    z_out = output
    cost = tf.reduce_mean(tf.square(output - theta))
    return {'x': x,'theta':theta, 'z': z_out,'W':encoder,'b':b_enc,'cost':cost}

def main_reg(Ytrain, Theta, dim = [30, 30, 30], ct = [.1, .1, .1], n_epochs = 100, batch_size=25,starter_learning_rate=1e-4,fname='deep_reg', optim = 'Adam', decay_after = 15, activation = 'leaky_relu', Ytest = None, Ttest = None):
    """
    This method is the main function to train the neural network.

    Parameters:
    -----------
    Ytrain: np.ndarray
        Input training data
    Theta: np.ndarray
        Label data for supervised learning
    dim: list of ints
        List of number of neurons per layer
    ct: list of floats
        List of values of the contamination level applied to the NN per layer
    n_epochs: int
        Number of epochs applied on the training.
    batch_size: int
        Batch size
    starter_learning_rate: float
        Initial learning rate.
    fname: str
        Name of the model.
    optim: str {'Adam', 'GD'}
        Optimizer (default 'Adam')
    decay_after: int
        How many epochs to wait until decay learning rate is applied.
    activation: str {'leaky_relu', 'relu_tanh', 'leaky_relu_l', 'tanh'}
        It defines the activation functions applied (default is 'leaky_relu')

            'leaky_relu':
                Leaky ReLU for all activation functions

            'relu_tanh':
                tanh on last hidden layer, Leaky ReLU for the rest

            'leaky_relu_l':
                Linear function on last hidden layer, Leaky ReLU for the rest

            'tanh':
                tanh for all activation functions
    Ytest: np.ndarray
        Input data for the test set.
    Ttest: np.ndarray
        Label for supervised trianing for the test set.

    Returns:
    reg: dict
        Trained model with:

            x: np.ndarray
                Input training data

            theta: np.ndarray
                Labels for supervised learning

            z: np.ndarray
                NN output

            W: np.ndarray
                Weights of NN

            b: np.ndarray
                Bias

            cost: float
                Value of cost function
    """

    n_samples=np.shape(Ytrain)[0]
    n_entries=np.shape(Ytrain)[1]
    reg = deep_reg(dimensions = dim, ct = ct, activation = activation)
    costs = np.zeros(n_epochs)
    costs_v = np.zeros(n_epochs)

    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    if optim == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(reg['cost'])
    elif optim == 'GD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(reg['cost'])
    else:
        print("ERROR: incorrect name of optimizer: " + optim)

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    print("\n\n")
    print("Fully supervised learning")
    print("\n\n")
    print("| Mean Sq error |   Epoch  |")
    print("|---------------|----------|")

    p_cost = np.array([0.])

    for epoch_i in range(n_epochs):

        # Pick one element to optimize for a given task

        if (epoch_i+1) >= decay_after:
            # epoch_n + 1 because learning rate is set for next epoch
            ratio = 1.0 * (n_epochs - (epoch_i+1))
            ratio = max(0, ratio / (n_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        for batch_i in range(n_samples // batch_size):

            # define train
            batch_xs = Ytrain[batch_i*batch_size:(batch_i+1)*batch_size,:]
            btheta = Theta[batch_i*batch_size:(batch_i+1)*batch_size,:]

            sess.run(optimizer, feed_dict={reg['x']: batch_xs,reg['theta']: btheta})

        cost = sess.run(reg['cost'], feed_dict={reg['x']: Ytrain,reg['theta']: Theta})
        costs[epoch_i] = cost

        if Ytest is not None and Ttest is not None:
            costv = sess.run(reg['cost'], feed_dict={reg['x']: Ytest,reg['theta']: Ttest})
            costs_v[epoch_i] = costv


        output = "|{0:13.6f}  |    {1}   |".format(cost,epoch_i)
        print(output)

    mdict=  {}

    mdict['layers'] = np.size(dim)
    mdict['W'] = sess.run(reg['W'], feed_dict={reg['x']: batch_xs})
    mdict['b'] = sess.run(reg['b'], feed_dict={reg['x']: batch_xs})
    mdict['cost'] = sess.run(reg['cost'], feed_dict={reg['x']: batch_xs,reg['theta']: btheta})
    mdict['costs'] = costs
    if Ytest is not None and Ttest is not None:
        mdict['costs_v'] = costs_v

    sio.savemat(fname+".mat",mdict)

    return reg

def encode_from_model(data, X, activation = 'leaky_relu'):
    """
    This method predicts the output from a model on a given data set.

    Parameters:
    -----------
    data: dict
        Data of ML model containing the weights of the model
    X: np.ndarray
        Data set for which we predict the output
    activation: str {'leaky_relu', 'relu_tanh', 'leaky_relu_l', 'tanh'}
        It defines the activation functions applied (default is 'leaky_relu')

            'leaky_relu':
                Leaky ReLU for all activation functions

            'relu_tanh':
                tanh on last hidden layer, Leaky ReLU for the rest

            'leaky_relu_l':
                Linear function on last hidden layer, Leaky ReLU for the rest

            'tanh':
                tanh for all activation functions

    Returns:
    --------
    output: np.ndarray
        Output model preduction from X data.
    P: list of np.ndarrays
        List of all outputs per layer
    """

    L = np.int(data['layers'])-1

    last_output = dp(X)
    W = data['W']
    be = data['b']

    all_h = []
    all_h.append(X)

    for layer_i in range(1,L+1):

        w = W[0,layer_i-1]
        b = be[0,layer_i-1].squeeze()

        if layer_i == L+1:
            output = np.dot(last_output, w) + b
        else:
            if activation == 'leaky_relu' or (activation in ['relu_tanh', 'leaky_relu_l'] and layer_i < L):
                output = np_leaky_relu(np.dot(last_output, w) + b)
            elif activation == 'leaky_relu_l' and layer_i == L:
                output = np.dot(last_output, w) + b
            elif activation == 'tanh' or (activation == 'relu_tanh' and layer_i == L):
                output = np.tanh(np.dot(last_output, w) + b)
        last_output = output
        all_h.append(output)

    P = {"all_h":all_h}

    return output,P

def ParamEstModel(fname='model',Xtrain=0,Xtest=0,Theta_test=0, version = 'deep_reg', activation = 'leaky_relu', top_fs = 0, v = False):
    """
    This method makes a parameter estimation of data set from a given model.

    Parameters:
    -----------
    fname: str
        Model name
    Xtrain: np.ndarray
        Training data set
    Xtest: np.ndarray
        Test set
    Theta_test: np.ndarray
        True parameters for test set
    version: str {'deep_reg', 'deep_reg_est', deep_regl', 'deep_regh', 'deep_regrh'}
        Version of learning code used (default is 'deep_reg')

            'deep_reg':
                Regression with Leaky ReLU as activation functions

            'deep_reg_est':
                Same as deep_reg but Ttrain, Ttest values centred to 1

            'deep_regl':
                Same as deep_reg_est but linear function on last hidden layer
                activation functions.

            'deep_regh':
                Same as deep_reg but tanh as activation functions

            'deep_regrh':
                Same as deep_reg but tanh on last hidden layer activation
                functions
    activation: str {'leaky_relu', 'relu_tanh', 'leaky_relu_l', 'tanh'}
        It defines the activation functions applied (default is 'leaky_relu')

            'leaky_relu':
                Leaky ReLU for all activation functions

            'relu_tanh':
                tanh on last hidden layer, Leaky ReLU for the rest

            'leaky_relu_l':
                Linear function on last hidden layer, Leaky ReLU for the rest

            'tanh':
                tanh for all activation functions
    top_fs: int
        It specifies how many feature space properties are used for the training,
        taking the most important ones. 0 if all are taken.
    v: bool
        Verbose, some text is shown if True

    Returns:
    --------
    Results: np.ndarray
        Parameter prediction on test set from ML model
    """
    if top_fs > 0:
        Xtrain = Xtrain[:,:top_fs] # TODO: test
        Xtest = Xtest[:,:top_fs] # TODO: test
    f = sio.loadmat(fname)

    z_mu,P = encode_from_model(f,Xtest, activation = activation)
    z_mut,P = encode_from_model(f,Xtrain, activation = activation)
    if v:
        print('estimating parameters from deep regressions')
    Results = {}
    if version in ['deep_reg_est', 'deep_regl']:
        Results["Pest"] = z_mu - 1.
    else:
        Results["Pest"] = z_mu
    Results["Error"] = np.mean((z_mu-Theta_test)**2,axis=1)

    return Results
