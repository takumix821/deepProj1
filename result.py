import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random

mnist = keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# input_mat 設定
train_size = train_images.shape[0]
test_size = test_images.shape[0]
input_mat = np.reshape(train_images, (train_size, train_images.shape[1]*train_images.shape[2]))
# test_images = np.reshape(test_images, (10000, 784))
input_mat.shape

# ymat 設定

ymat = np.repeat(0, 10*train_size).reshape(train_size, 10)

for i in range(train_size):
    ymat[i, train_labels[i]] = 1

def sigmoid(x):
    return 1/(1 + np.exp(-x))

random.seed(0)

# 參數設定
learning_rate = 0.2

#input_mat = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #
#ymat = np.array([[0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]]) #

matrix_num = 40
n_in = input_mat.shape[1]
n_hidden = [matrix_num, matrix_num, matrix_num , matrix_num] # 隱藏層兩層 各有?, ?個neuron
n_out = ymat.shape[1]

# 第1層權重初始值
w = np.random.randn(n_hidden[0], n_in)

# 第2層權重初始值
w2 = np.random.randn(n_hidden[1], n_hidden[0])

# 第3層權重初始值
w3 = np.random.randn(n_hidden[2], n_hidden[1])

# 第4層權重初始值
w4 = np.random.randn(n_hidden[3], n_hidden[2])

# 第5層權重初始值
w5 = np.random.randn(n_out, n_hidden[3])

t = 0


while True:
    
    input_ = input_mat[t % input_mat.shape[0]]
    y = ymat[t % input_mat.shape[0]]

    ### 正向傳播

    # 計算output
    hidden_1 = sigmoid(np.matmul(w, input_))
    hidden_2 = sigmoid(np.matmul(w2, hidden_1))
    hidden_3 = sigmoid(np.matmul(w3, hidden_2))
    hidden_4 = sigmoid(np.matmul(w4, hidden_3))
    output_ = sigmoid(np.matmul(w5, hidden_4))



    ### 誤差反向傳播

    # 第4層權重delta計算
    delta5 = (y - output_) * output_ * (1 - output_)

    # 第3層權重delta計算
    delta4 = hidden_4 * (1 - hidden_4) * np.matmul(delta5, w5)

    # 第3層權重delta計算
    delta3 = hidden_3 * (1 - hidden_3) * np.matmul(delta4, w4)

    # 第2層權重delta計算
    delta2 = hidden_2 * (1 - hidden_2) * np.matmul(delta3, w3)

    # 第1層權重delta計算
    delta1 = hidden_1 * (1 - hidden_1) * np.matmul(delta2, w2)

    # 第5層權重修正
    w5 += learning_rate * np.tensordot(delta5, hidden_4, axes = 0)
    
    # 第4層權重修正
    w4 += learning_rate * np.tensordot(delta4, hidden_3, axes = 0)

    # 第3層權重修正
    w3 += learning_rate * np.tensordot(delta3, hidden_2, axes = 0)

    # 第2層權重修正
    w2 += learning_rate * np.tensordot(delta2, hidden_1, axes = 0)

    # 第1層權重修正
    w += learning_rate * np.tensordot(delta1, input_, axes = 0)

    t += 1

    if t > 10000 or (abs(output_ - y) < 0.000001 ).all():
        print('t=',t)
        break


test_mat = np.reshape(test_images, (10000, 784)).T
test_hidden_1 = sigmoid(np.matmul(w, test_mat))
test_hidden_2 = sigmoid(np.matmul(w2, test_hidden_1))
test_hidden_3 = sigmoid(np.matmul(w3, test_hidden_2))
test_hidden_4 = sigmoid(np.matmul(w4, test_hidden_3))
test_output_ = sigmoid(np.matmul(w5, test_hidden_4))


output_y = np.zeros(test_size)

for i in range(test_size):
    for j in range(10):
        if test_output_.T[i, j] >= max(test_output_.T[i, ]):
            output_y[i] = j

#output_raw = np.round_(test_output_.T)
#class_ = np.arange(10)
#output_y = np.matmul(output_raw, class_)

print(output_y)

print(sum(test_labels == output_y)/test_size)

