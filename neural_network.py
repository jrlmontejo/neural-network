#!/usr/bin/env python3

"""Neural Network

Compiler: Python 3.6.5
OS: macOS 10.13.6
"""

import numpy as np

INPUT_LAYER_SIZE = 354
HIDDEN1_LAYER_SIZE = 5
HIDDEN2_LAYER_SIZE = 4
OUTPUT_LAYER_SIZE = 3

def binarize_labels(labels):
  """Converts the labels into their binary representation

  The labels consist of 8 classes (1, 2, ..., 8) and can be mapped
  to binaries (000, 001, ..., 111) to represent the 3 neurons in the
  output layer:

  Class 1 => 000
  Class 2 => 001
  ...
  Class 8 => 111

  Parameters:
    labels: The array of labels for each training example

  Returns:
    The array of binary vectors
  """

  y = []
  for label in labels:
    l = int(label) - 1
    vec = list(np.binary_repr(l, width=3))
    vec = [int(b) for b in vec]
    y.append(vec)
  return y

def partition_dataset(X, Y):
  labels = binarize_labels(Y)

  # split 70 / 30
  N = int(len(X) * 0.7)
  training_data, validation_data = np.split(X, [ N ])
  training_labels, validation_labels = np.split(labels, [ N ])

  return (training_data, training_labels), (validation_data, validation_labels)

def randomize(size):
  """Returns samples from a uniform distribution with range -0.1 to 0.1

  Parameters:
    size: The number of samples to generate. If size is (m, n), then m * n samples are generated

  Returns:
    The array of generated samples
  """

  return np.random.uniform(-0.1, 0.1, size)

def init_weights():
  """Initializes the weights of the neurons of each layer in the network

  Returns:
    The initial weights grouped per layer
  """

  h1 = randomize((HIDDEN1_LAYER_SIZE, INPUT_LAYER_SIZE))
  h2 = randomize((HIDDEN2_LAYER_SIZE, HIDDEN1_LAYER_SIZE))
  out = randomize((OUTPUT_LAYER_SIZE, HIDDEN2_LAYER_SIZE))

  return { 'h1': h1, 'h2': h2, 'out': out }

def init_biases():
  """Initializes the biases of the neurons of each layer in the network

  Returns:
    The initial biases grouped per layer
  """

  h1 = randomize(HIDDEN1_LAYER_SIZE)
  h2 = randomize(HIDDEN2_LAYER_SIZE)
  out = randomize(OUTPUT_LAYER_SIZE)

  return { 'h1': h1, 'h2': h2, 'out': out }

def weighted_sum(x, w, b):
  """Returns the weighted sums of the inputs

  Parameters:
    x: The array of inputs
    w: The array of weights of the inputs
    b: The array of biases

  Returns:
    The array of weighted sums
  """

  return np.dot(w, x) + b

def sigmoid(v):
  """Returns the result of the sigmoid activation function

  Parameters:
    v: The array of weighted sums

  Returns:
    The array of sigmoid outputs
  """

  return 1.0 / (1 + np.exp(-v))

def feed_forward(x, weights, biases):
  """Performs a forward pass from the input layer to the output layer

  Parameters:
    x: The array of inputs
    weights: The array of weights grouped per layer
    biases: The array of biases grouped per layer

  Returns:
    The outputs of the neurons for each layer
  """

  v_h1 = weighted_sum(x, weights['h1'], biases['h1'])
  y_h1 = sigmoid(v_h1)

  v_h2 = weighted_sum(y_h1, weights['h2'], biases['h2'])
  y_h2 = sigmoid(v_h2)

  v_out = weighted_sum(y_h2, weights['out'], biases['out'])
  y_out = sigmoid(v_out)

  return { 'h1': y_h1, 'h2': y_h2, 'out': y_out }

def delta(*args):
  """Returns the local gradient for one layer

  If the parameters are the outputs and the error, the local gradients in the
  output layer will be computed. If the parameters are the outputs, the local
  gradients from the previous neurons and the weights, the local gradients in
  the hidden layer will be computed.

  Returns:
    The array of local gradients for the layer
  """

  if len(args) == 3:
    y, prev_delta, weights = args
    delta = y * (1 - y) * np.dot(prev_delta, weights)
  elif len(args) == 2:
    y, error = args
    delta = error * y * (1 - y)
  return delta

def reshape_delta(delta):
  """Reshapes the local gradient array into 2d to be used for calculating weights

  Parameters:
    delta: The array of local gradients to be reshaped

  Returns:
    The 2d array of local gradients
  """

  return np.reshape(delta, (len(delta), -1))

def compute_local_gradients(y, error, weights):
  """Computes the local gradients for all layers

  Parameters:
    y: The array of outputs per layer
    error: The error of each neuron in the output layer
    weights: The array of weights per layer

  Returns:
    dictionary
      The array of local gradients for each layer
  """

  delta_out = delta(y['out'], error)
  delta_h2 = delta(y['h2'], delta_out, weights['out'])
  delta_h1 = delta(y['h1'], delta_h2, weights['h2'])

  return { 'out': delta_out, 'h2': delta_h2, 'h1': delta_h1 }

def update_weights(weights, deltas, y, x, eta):
  """Updates the weights

  Parameters:
    weights: The array of current weights
    deltas: The array of local gradients per layer
    y: The array of outputs per layer
    x: The array of inputs
    eta: The learning rate

  Returns:
    The array of updated weights grouped per layer
  """

  out = weights['out'] + (eta * reshape_delta(deltas['out']) * y['h2'])
  h2 = weights['h2'] + (eta * reshape_delta(deltas['h2']) * y['h1'])
  h1 = weights['h1'] + (eta * reshape_delta(deltas['h1']) * x)

  return { 'out': out, 'h2': h2, 'h1': h1 }

def update_biases(biases, deltas, eta):
  """Updates the weights

  Parameters:
    weights: The array of current weights
    deltas: The array of local gradients per layer
    eta: The learning rate

  Returns:
    The array of updated biases grouped per layer
  """

  out = biases['out'] + (eta * deltas['out'])
  h2 = biases['h2'] + (eta * deltas['h2'])
  h1 = biases['h1'] + (eta * deltas['h1'])

  return { 'out': out, 'h2': h2, 'h1': h1 }

def cost(errors):
  """Computes the average error for each training epoch

  Parameters:
    errors: The errors from all training examples

  Returns:
    The average error
  """

  total_errors = [np.sum(e * e) * 0.5 for e in errors]
  return np.sum(total_errors) / len(total_errors)

def shuffle_indices(n):
  """Generates an array of numbers from 0 to n then shuffles it

  This is used for shuffling the data set before training.

  Parameters:
    n: The end point of the array of numbers

  Returns:
    Shuffled array of numbers
  """

  p = np.arange(n)
  np.random.shuffle(p)
  return p

def train(eta=0.1, epochs=30000):
  """Trains the network

  Parameters:
    eta: The learning rate
    epochs: The number of iterations
  """

  training_data = np.genfromtxt('training_set.csv', delimiter=',')
  training_labels = binarize_labels(np.genfromtxt('training_labels.csv', delimiter=','))
  validation_data = np.genfromtxt('validation_set.csv', delimiter=',')
  validation_labels = binarize_labels(np.genfromtxt('validation_labels.csv', delimiter=','))

  training_size = len(training_labels)
  validation_size = len(validation_labels)

  # initialize weights & biases
  weights = init_weights()
  biases = init_biases()

  for epoch in range(0, epochs):
    training_errors = []
    validation_errors = []

    p = shuffle_indices(training_size)
    q = shuffle_indices(validation_size)

    # train
    for n in range(0, training_size):
      i = p[n]
      x = training_data[i]
      d = training_labels[i]
      y = feed_forward(x, weights, biases)

      # back propagate
      error = d - y['out']
      deltas = compute_local_gradients(y, error, weights)
      weights = update_weights(weights, deltas, y, x, eta)
      biases = update_biases(biases, deltas, eta)
      training_errors.append(error)

    # validate
    for n in range(0, validation_size):
      i = q[n]
      x = validation_data[i]
      d = validation_labels[i]
      y = feed_forward(x, weights, biases)
      error = d - y['out']
      validation_errors.append(error)

    training_error = cost(training_errors)
    validation_error = cost(validation_errors)

    print('Iteration: {} Training Error: {} Validation Error: {}'.format(epoch, training_error, validation_error))
    if validation_error < 0.001:
      break

  print('Total epochs: {}'.format(epoch))
  print('Training error at termination: {}'.format(training_error))
  print('Validation error at termination: {}'.format(validation_error))

# def test():
#   DATA = np.genfromtxt('test_set.csv', delimiter=',')


train()
