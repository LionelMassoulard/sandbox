# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 11:36:43 2021

@author: LionelMassoulard
"""



import scipy.stats as sp
import numpy as np

np.random.seed(123)

import matplotlib.pylab as plt

def mu(x):
    return 6-2*x+x**2

def sigma(x):
    return 0.5+np.log(1+2*x**2)


def get_data(batch=4048):
    
    X = np.random.randn(batch, 1).astype(np.float32)
    # oo = np.argsort(X[:, 0])
    # X = X[:, oo]
    
    n = np.random.randn(batch, 1).astype(np.float32)

    m = mu(X)
    s = sigma(X)
    y = m + s * n

    norm_quantiles = sp.norm.ppf(quantiles_list)[np.newaxis, :]
    
    
    return X, y, m, s, m + s * norm_quantiles

X, y , m, s, q = get_data(4048)

plt.cla()
plt.plot(X[:, 0], y[:, 0], ".", markersize=1)



import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(5, activation="linear"))


y_pred = model(X)
y_true = y #y, dtype=tf.float32)

quantiles_list = [0.1, 0.25, 0.5, 0.75, 0.9]
def loss(y_true, y_pred):
    quantiles = tf.constant([quantiles_list])
    error = y_true - y_pred
    l = tf.reduce_sum(tf.maximum(quantiles*error, (quantiles-1)*error), axis=1)
    return l

model.compile(optimizer="rmsprop", loss=loss)#♣"mse")
model.fit(X, y_true, batch_size=128, epochs=100)

X_test, y_test , m_test, s_test, q_test = get_data(1024)
oo = np.argsort(X_test[:, 0])

y_pred = model(X_test).numpy()


# %matplotlib qt

plt.figure(0, figsize=(10, 10))
plt.cla()
plt.plot(X_test[oo, 0], y_test[oo, 0], ".", markersize=1)
for j in range(5):
    plt.plot(X_test[oo, 0], y_pred[oo, j], label=f"q {quantiles_list[j]*100}")
    plt.plot(X_test[oo, 0], q_test[oo, j], label=f"q {quantiles_list[j]*100} True")
plt.legend()


# In[]
import tensorflow_probability as tfp


    
    