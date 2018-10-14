import numpy as np
from imblearn.combine import SMOTEENN
from collections import Counter

X = np.genfromtxt('data.csv', delimiter=',')
Y = np.genfromtxt('data_labels.csv', delimiter=',')

N = len(Y)

# shuffle dataset
indices = np.arange(N)
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# partition dataset into training and validation sets
# split 70 / 30
split = int(N * 0.7)
training_set, training_labels = X[:split], Y[:split]
validation_set, validation_labels = X[split:], Y[split:]

# resample training set
res_training_set, res_training_labels = SMOTEENN().fit_resample(training_set, training_labels)
TN = len(res_training_labels)
indices = np.arange(TN)
np.random.shuffle(indices)
res_training_set = res_training_set[indices]
res_training_labels = res_training_labels[indices]

print('Before resampling: {}'.format(Counter(training_labels)))
print('After resampling: {}'.format(Counter(res_training_labels)))

np.savetxt('training_set.csv', res_training_set, delimiter=',', fmt='%f')
np.savetxt('training_labels.csv', res_training_labels, delimiter=',', fmt='%i')
np.savetxt('validation_set.csv', validation_set, delimiter=',', fmt='%f')
np.savetxt('validation_labels.csv', validation_labels, delimiter=',', fmt='%i')
