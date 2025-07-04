from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl


def build_dual_output_model(input_dim=None, latent_dim=None, output_timesteps=None,
                            dispV_dim=None, dispfield_dim1=None, dispfield_dim2=None):
    # Input layer
    inputs = layers.Input(shape=(input_dim,))
    
    # Dense encoder
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(latent_dim, activation='relu')(x)
    
    # Repeat vector for LSTM
    x = layers.RepeatVector(output_timesteps)(x)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    
    # Branch 0: 1D displacement vector
    x0 = layers.TimeDistributed(layers.Dense(10 * 32, activation='relu'))(x)
    x0 = layers.TimeDistributed(layers.Reshape((10, 32)))(x0)
    x0 = layers.TimeDistributed(layers.Conv1DTranspose(1, kernel_size=3, strides=2, activation='relu'))(x0)
    disp_output = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='dispVector_output')(x0)
    
    # Branch 1: 2D displacement field
    x1 = layers.TimeDistributed(layers.Dense(dispfield_dim1 * dispfield_dim2 * latent_dim, activation='relu'))(x)
    x1 = layers.TimeDistributed(layers.Reshape((dispfield_dim1, dispfield_dim2, latent_dim)))(x1)
    x1 = layers.TimeDistributed(layers.Conv2DTranspose(64, 3, strides=(1,1), padding='same', activation='relu'))(x1)
    x1 = layers.TimeDistributed(layers.Conv2DTranspose(32, 3, strides=(1,1), padding='same', activation='relu'))(x1)
    x1 = layers.TimeDistributed(layers.Conv2DTranspose(16, 3, strides=(1,1), padding='same', activation='relu'))(x1)
    x1 = layers.TimeDistributed(layers.Conv2DTranspose(1, 3, strides=(1,1), padding='same', activation='linear'))(x1)
    dispfield_output = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='dispField_output')(x1)
    
    # Build and compile model
    model = models.Model(inputs=inputs, outputs=[disp_output, dispfield_output])
    model.compile(optimizer='adam',
                  loss=['mse', 'mse'],
                  loss_weights=[1.0, 1.0])
    
    return model
