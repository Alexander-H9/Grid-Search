import itertools
import numpy as np
import tensorflow as tf


batch_size = [4, 8]
neurons = [16, 32]
seed = [1,2,3]
layers = [3, 5]

losses = {}


def train(hyper):
        (batch, neurons, seed, layers) = hyper
        
        np.random.seed(seed)
        python_random.seed(seed)
        tf.random.set_seed(seed)
        initializer = tf.keras.initializers.HeNormal(seed=seed)
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(1,)))
        for i in range(layers):
            model.add(tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer=initializer))
        model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer))
            
        model.compile(loss='mse', optimizer='adam'),
        history = model.fit(x_scaled, y_scaled, epochs=50000, batch_size=batch, verbose=0, callbacks=[callback])
        loss_eval = model.evaluate(x_scaled, y_scaled, batch_size=128, verbose=0)
        
        return history.history['loss'][-1], model


for parameterset in list(itertools.product(batch_size, neurons, seed, layers)):
        loss, model = train(parameterset)
        print(f"Loss {parameterset}: {loss}")
        losses[parameterset] = loss
        if loss < min_loss:
            min_loss = loss
            best_model = model
