
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

celsiul_q = np.array([-40,-10,0,8,15,22,38], dtype=float)
farenheit_a = np.array([-40,14,32,46,59,72,100], dtype=float)

for i, c in enumerate(celsiul_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, farenheit_a[i]))

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4, input_shape=[1])
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0,l1,l2])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsiul_q, farenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0,24]))

print("These are the layer variables: {}". format(l0.get_weights()))
print("These are the layer variables: {}". format(l1.get_weights()))
print("These are the layer variables: {}". format(l2.get_weights()))


