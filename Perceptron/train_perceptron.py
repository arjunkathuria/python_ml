import pandas as pd
import numpy as np
from perceptron import Perceptron
import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('../datasets/iris.data', header=None)

# print(df.tail())

Y = df.iloc[0:100, 4].values

# only training for 2 classes atm
Y = np.where(Y == 'Iris-setosa', 1, -1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50: 100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, Y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Number of Misclassifications')

print('Errors: ', ppn.errors_)
print('see the lovely converging graph now : )')

plt.show()
