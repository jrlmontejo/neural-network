import numpy as np

a = np.array([[1, 1, 1], [0, 0, 1]])
b = [int(''.join(i), 2) + 1 for i in a.astype(str)]
print(b)

# a_s = [str(i) for i in a]
# b = ''.join(a_s)
# c = int(b, 2)
# print(c)

# a = [[0.51, 0.9822, 0.87]]
# binarized_labels = np.around(a).astype(int)
# print(binarized_labels)

# labels = np.array(a).astype(str)
# print(''.join(labels))

np.