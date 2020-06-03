"""
DESCRIPTION
This script allows to make a regression from a deep neural network.

"""

import numpy as np
import tensorflow as tf
import scipy.io as sio
from copy import deepcopy as dp

def leaky_relu(z, name=None):
  """
  It operates a Leaky ReLU on the value z.
  Input:
  z: value to which we operate.
  name: a name for the operation (optional).
  Output:
  Result of Leaky ReLU operation.
  """
  return tf.maximum(0.01 * z, z, name=name)

def np_leaky_relu(z):
  """
  Numpy version of Leaky ReLU.
  Input:
  z: value to which we operate.
  Output:
  Result of Leaky ReLU operation.
  """
  return np.maximum(0.01 * z, z)

def deep_reg(dimensions=[784, 512, 256, 64],ct=[0.1,0.1,0.1,0.1], activation = 'leaky_relu'):
    """
    It builds the architecture corresponding to a NN regression.
    Input:
    dimensions: List of number of neurons per layer.
    ct: list of values of the contamination level applied to the NN per layer.
    activation: it defines the activation functions applied. Options can be:
    'leaky_relu', 'relu_tanh', 'leaky_relu_l', 'tanh', 'relu_tanh'.

    Output:
    Dictionary with:
    x: input training data.
    theta: labels for supervised learning.
    z: NN output.
    W: weights of NN.
    b: bias.
    cost: value of cost function.
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
    Main function to train the neural network.
    Input:
    Ytrain: input training data
    Theta: label data for supervised learning
    dim: list of number of neurons per layer.
    ct: list of values of the contamination level applied to the NN per layer.
    n_epochs: number of epochs applied on the training.
    batch_size: batch size
    starter_learning_rate: initial learning rate.
    fname: name of the model.
    optim: optimizer. It can be 'Adam' or 'GD'.
    decay_after: how many epochs to wait until decay learning rate is applied.
    activation: it defines the activation functions applied (see deep_reg()).
    Ytest: input data for the test set.
    Ttest: label for supervised trianing for the test set.

    Output:
    reg: trained model with output corresponding to deep_reg().
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
    It predicts the output from a model on a given data set.
    Input:
    data: data of ML model containing the weights of the model.
    X: data set for which we predict the output.
    activation: it defines the activation functions applied (see deep_reg()).

    Output:
    output: output model preduction from X data.
    P: list of all outputs per layer.
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
    Parameter estimation of data set from a given model.

    Input:
    fname: model name.
    Xtrain: training data set.
    Xtest: test set.
    Theta_test: true parameters for test set.
    version: version of learning code used. Can be 'deep_reg', 'deep_reg_est', deep_regl', 'deep_regh', 'deep_regrh'
    activation: it defines the activation functions applied (see deep_reg()).
    top_fs: Specifies how many feature space properties are used for the training, taking the most important ones. 0 if all are taken.
    v: verbose.

    Output:
    Results: parameter prediction on test set from ML model. 
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
