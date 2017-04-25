import time
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from common.functions import *
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


mnist_raw = loadmat("./mnist-original.mat")
mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
        }
print("Success!")



X=mnist["data"]

Y=mnist["target"]
x_train, x_test, y_train_label, y_test_label = train_test_split(X, Y, test_size=0.1)

x_train = x_train.astype(np.float32)
x_train = x_train / 255.0
x_test = x_test.astype(np.float32)
x_test = x_test / 255.0


y_train = preprocessing.LabelBinarizer().fit_transform(y_train_label)
y_test = preprocessing.LabelBinarizer().fit_transform(y_test_label)

from common.multi_layer_net import MultiLayerNet
from common.optimizer import *

weight_decay_lambda = 0
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100], output_size=10,
                        activation = "sigmoid",weight_decay_lambda=weight_decay_lambda)
optimizer = Adam()

max_epochs = 30
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    start = time.time()
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        end = time.time() - start

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + ", elapsed:{}".format(end) + "[sec]")

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
