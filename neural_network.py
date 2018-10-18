#!/usr/bin/env python3

"""Neural Network

Compiler: Python 3.6.5
OS: macOS 10.13.6
"""

import numpy as np
import time

# network architecture
INPUT_LAYER_SIZE = 354
HIDDEN1_LAYER_SIZE = 9
HIDDEN2_LAYER_SIZE = 8
OUTPUT_LAYER_SIZE = 3

# number of epochs before early stopping if there
# is no decrease in validation set error
PATIENCE = 35

def binarize(labels):
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

def decimalize(labels):
  """Converts the binary vectors back to their corresponding labels

  000 => Class 1
  001 => Class 2
  ...
  111 => Class 8

  Parameters:
    labels: The array of binarized labels

  Returns:
    The array of labels
  """

  label_values = [int(''.join(i), 2) + 1 for i in labels.astype(str)]
  return label_values

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

def error(y, d):
  """Returns the error in the output layer

  Parameters:
    y: the response from the output layer
    d: the desired or expected output

  Returns:
    The error in the output layer
  """

  return d - y

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

def train(training_set, validation_set, eta=0.1, epochs=10000):
  """Trains the network

  Parameters:
    training_set: The training set and labels
    validation_set: The validation set and labels
    eta: The learning rate
    epochs: The number of iterations

  Returns:
    The final weights and biases
  """

  training_data, training_labels = training_set
  validation_data, validation_labels = validation_set

  training_size = len(training_labels)
  validation_size = len(validation_labels)

  # initialize weights & biases
  weights = init_weights()
  biases = init_biases()

  # patience counter
  j = 0

  # storage for the best parameters so far
  best_params = {
    'training_error': 1.0,
    'validation_error': 1.0,
    'epoch': 0,
    'weights': weights,
    'biases': biases
  }

  start = time.time()

  for epoch in range(0, epochs):
    training_errors = []
    validation_errors = []

    p = shuffle_indices(training_size)
    q = shuffle_indices(validation_size)

    # training phase
    for n in range(0, training_size):
      i = p[n]
      x = training_data[i]
      d = training_labels[i]
      y = feed_forward(x, weights, biases)

      # back propagate
      err = error(y['out'], d)
      deltas = compute_local_gradients(y, err, weights)
      weights = update_weights(weights, deltas, y, x, eta)
      biases = update_biases(biases, deltas, eta)
      training_errors.append(err)

    # validation phase
    for n in range(0, validation_size):
      i = q[n]
      x = validation_data[i]
      d = validation_labels[i]
      y = feed_forward(x, weights, biases)
      err = error(y['out'], d)
      validation_errors.append(err)

    training_error = cost(training_errors)
    validation_error = cost(validation_errors)

    print('Iteration: {} Training Error: {} Validation Error: {}'.format(epoch + 1, training_error, validation_error))

    # save the model parameters if there is a decrease in validation error
    if validation_error < best_params['validation_error']:
      best_params['training_error'] = training_error
      best_params['validation_error'] = validation_error
      best_params['epoch'] = epoch
      best_params['weights'] = weights
      best_params['biases'] = biases
      j = 0
    else:
      j = j + 1

    # early stopping
    if j > PATIENCE:
      break

  end = time.time()

  print('\nTraining time: {} seconds'.format(end - start))

  print('\nTotal epochs: {}'.format(epoch + 1))
  print('Training error at termination: {}'.format(training_error))
  print('Validation error at termination: {}'.format(validation_error))

  print('\nChosen iteration: {}'.format(best_params['epoch'] + 1))
  print('Training error at chosen iteration: {}'.format(best_params['training_error']))
  print('Validation error at chosen iteration: {}'.format(best_params['validation_error']))

  return best_params['weights'], best_params['biases']

def test(test_data, weights, biases):
  """Tests the network on the test set

  Parameters:
    test_data: The test data set
    weights: The weights of the network per layer
    biases: The biases of the network per layer

  Returns:
    The array of prediction results
  """

  results = []

  for x in test_data:
    y = feed_forward(x, weights, biases)
    labels = np.around(y['out']).astype(int)
    results.append(labels)

  return decimalize(np.array(results))

def start():
  training_set_file = 'training_set.csv'
  training_labels_file = 'training_labels.csv'

  validation_set_file = 'validation_set.csv'
  validation_labels_file = 'validation_labels.csv'

  print('Fetching training data from {}...'.format(training_set_file))
  training_data = np.genfromtxt(training_set_file, delimiter=',')

  print('Fetching training labels from {}...'.format(training_labels_file))
  training_labels = binarize(np.genfromtxt(training_labels_file, delimiter=','))

  print('Fetching validation data from {}...'.format(validation_set_file))
  validation_data = np.genfromtxt(validation_set_file, delimiter=',')

  print('Fetching validation labels from {}...'.format(validation_labels_file))
  validation_labels = binarize(np.genfromtxt(validation_labels_file, delimiter=','))

  # train network
  weights, biases = train((training_data, training_labels), (validation_data, validation_labels))

  test_set_file = 'test_set.csv'

  print('Fetching test data from {}...'.format(test_set_file))
  test_data = np.genfromtxt(test_set_file, delimiter=',')

  # test model on the test set
  results = test(test_data, weights, biases)

  # save predictions to file
  predicted_file = 'predicted_ann4.csv'

  print('Saving predictions to {}...'.format(predicted_file))
  np.savetxt(predicted_file, results, delimiter=',', fmt='%i')
  print('Test results saved to {}'.format(predicted_file))

start()


