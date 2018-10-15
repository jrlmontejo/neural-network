import matplotlib.pyplot as plt
import numpy as np

training_errors = np.genfromtxt('training_errors.csv', delimiter=',')
validation_errors = np.genfromtxt('validation_errors.csv', delimiter=',')
epochs = np.arange(len(training_errors))

plt.plot(epochs, training_errors, label='Training')
plt.plot(epochs, validation_errors, label='Validation')
plt.axvline(x=244, linestyle='--', color='r', linewidth=1, label='Best')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.show()