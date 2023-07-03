import tensorflow as tf 
import numpy as np


def valina_fc_ae():
    """
    create a fully-convolutional autoencoder
    Input: 500x500x3
    Output: 500x500x3
    """
    x = tf.keras.layers.Input(shape=(500, 500,3))
    conv1 = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    conv1 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), activation='relu', padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu', padding='same')(conv2)
    conv3 = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), activation='relu', padding='same')(conv2)
    conv3 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same')(conv3)
    latent  = tf.keras.layers.Conv2D(512, (3,3), strides=(2,2), activation=None, padding='same')(conv3)
    dconv1 = tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2,2), activation='relu', padding='same')(latent)
    dconv1 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same')(dconv1)
    dconv2 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), activation='relu', padding='same')(dconv1)
    dconv2 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu', padding='same')(dconv2)
    dconv3 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same')(dconv2)
    dconv3 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same')(dconv3)
    out = tf.keras.layers.Conv2DTranspose(3, (3,3), strides=(2,2), activation='relu', padding='same')(dconv3)
    
    return tf.keras.Model(x, out)

def main():
    model = valina_fc_ae()
    print(model.summary())
    
if __name__ == "__main__":
    main()