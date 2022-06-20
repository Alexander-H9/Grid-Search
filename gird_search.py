import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras


batch_size = [4, 8]
neurons = [16, 32]
seed = [1,2]
layers = [3, 5]

min_loss = 1.0
losses = {}
x_scaled = np.array([[1,1],[2,0],[0,2],[123,123],[100,10]])
y_scaled = np.array([[2],[2],[2],[246],[110]])

x_val = np.array([[2,3],[4,40],])
y_val = np.array([[5],[44],])


def train(hyper):

        (batch, neurons, seed, layers) = hyper
        np.random.seed(seed)
        #python_random.seed(seed)
        tf.random.set_seed(seed)
        initializer = tf.keras.initializers.HeNormal(seed=seed)
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(2,)))
        for i in range(layers):
            model.add(tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer=initializer))
        model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer))
            
        model.compile(loss='mse', optimizer='adam'),
        history = model.fit(x_scaled, y_scaled, epochs=3000, batch_size=batch, verbose=0) # callbacks=
        loss_eval = model.evaluate(x_scaled, y_scaled, batch_size=128, verbose=0)
        
        return history.history['loss'][-1], model


for parameterset in list(itertools.product(batch_size, neurons, seed, layers)):
        loss, model = train(parameterset)
        print(f"Loss {parameterset}: {loss}")
        losses[parameterset] = loss
        if loss < min_loss:
            min_loss = loss
            best_model = model

print("losses:", losses)
print("best model", best_model)

res = best_model.predict(np.array([[1,7], [1,1], [25, 18]]))
print("res: ", res)