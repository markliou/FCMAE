import tensorflow as tf 
import numpy as np

def concept_gated_conv_ae():
    """
    create a gated convolutional autoencoder
    Input: 256x256x3
    Output: 256x256x3
    """
    x = tf.keras.layers.Input(shape=(256, 256, 3))
    
    # pixel embedding
    enc = tf.keras.layers.Conv2D(128, (1,1), activation=mish)(x)
    
    # position embedding
    pos = tf.random.uniform(shape=(256, 256, 1), minval=-1., maxval=1.)
    emb = tf.concat([enc, pos], axis=-1)
    
    
    
    
    out = None
    return tf.keras.Model(x, out)
    pass

def concept_gated_conv(x):
    # extract concept
    
    # concept injection
    pass

def concept_extract_conv(x, embedding):
    concept3 = concept_gated_conv(x, 3, embedding)
    concept5 = concept_gated_conv(x, 5, embedding)
    concept7 = concept_gated_conv(x, 7, embedding)
    pass

def concept_injection_conv(x):
    pass

def concept_gated_conv(x, kernel_size, embedding):
    gate = tf.keras.layers.Conv2D(embedding, (kernel_size, kernel_size), activation=tf.keras.activations.sigmoid)(x)
    enc = tf.keras.layers.Conv2D(embedding, (1, 1), activation=None)(x)
    return gate * enc
    pass

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def main():
    pass


if __name__ == "__main__":
    main()