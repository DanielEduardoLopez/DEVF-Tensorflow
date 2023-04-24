# Simple Neural Network
# Daniel Eduardo López
# 17/04/2023

# Libraries importation
import numpy as np
import pandas as pd
import tensorflow as tf


def model_sol():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-4.0, 1.0, 6.0, 11.0, 16.0, 21.0], dtype=float)

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=[1])]
    )

    model.compile(loss= 'mean_squared_error', optimizer = 'sgd')
    model.fit(xs, ys, epochs = 500)

    print(model.predict([10.0]))

    return model

if __name__ == '__main__':
    model = model_sol()
    model.save('mymodel.h5')