import tensorflow as tf 
import numpy as np


def valina_fc_ae():
    """
    create a fully-convolutional autoencoder
    Input: 256x256x3
    Output: 256x256x3
    """
    x = tf.keras.layers.Input(shape=(256, 256, 3))
    conv1 = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation=mish, padding='same')(x)
    conv1 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation=mish, padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), activation=mish, padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation=mish, padding='same')(conv2)
    conv3 = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), activation=mish, padding='same')(conv2)
    conv3 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation=mish, padding='same')(conv3)
    latent  = tf.keras.layers.Conv2D(512, (3,3), strides=(2,2), activation=None, padding='same')(conv3)
    dconv1 = tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2,2), activation=mish, padding='same')(latent)
    dconv1 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation=mish, padding='same')(dconv1)
    dconv2 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), activation=mish, padding='same')(dconv1)
    dconv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation=mish, padding='same')(dconv2)
    dconv3 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), activation=mish, padding='same')(dconv2)
    dconv3 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation=mish, padding='same')(dconv3)
    out = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), activation=mish, padding='same')(dconv3)
    out = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation=mish, padding='same')(out)
    out = tf.keras.layers.Conv2D(3, (3,3), strides=(1,1), activation=None, padding='same')(out)
    
    return tf.keras.Model(x, out)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def main():
    model = valina_fc_ae()
    print(model.summary())
    print(model(tf.ones([5, 256, 256,3])))
    
if __name__ == "__main__":
    main()