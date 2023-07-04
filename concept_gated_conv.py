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
    
    # concept convolution
    conv1 = tf.keras.layers.LayerNormalization()(emb)
    conv1 = concept_gated_conv(conv1, 64)
    conv1 = tf.keras.layers.Conv2D(64, (1,1), activation=mish)(conv1)
    conv1 = tf.keras.layers.Conv2D(64, (1,1), activation=mish)(conv1)
    conv1 = tf.keras.layers.Conv2D(64, (1,1), activation=mish)(conv1)
    
    
    
    out = None
    return tf.keras.Model(x, out)
    pass

def concept_gated_conv(x, embedding):
    # extract concept
    concept = concept_extract_conv(x, embedding)
    # concept injection
    feature = concept_injection_conv(x, concept, embedding)
    return feature
    pass

def concept_extract_conv(x, embedding):
    blank = tf.zeros_like(x)
    concept3 = concept_gated_conv(x, blank, 3, embedding)
    concept5 = concept_gated_conv(x, blank, 5, embedding)
    concept7 = concept_gated_conv(x, blank, 7, embedding)
    concept = concept3 + concept5 + concept7
    return tf.math.reduce_sum(concept, axis=[1, 2])
    pass

def concept_injection_conv(x, concept, embedding):
    feature3 = concept_gated_conv(x, concept, 3, embedding)
    feature5 = concept_gated_conv(x, concept, 5, embedding)
    feature7 = concept_gated_conv(x, concept, 7, embedding)
    return feature3 + feature5 + feature7
    pass

def concept_gated_conv(x, concept, kernel_size, embedding):
    gate = tf.keras.layers.Conv2D(embedding, (kernel_size, kernel_size), activation=tf.keras.activations.sigmoid)(x)
    enc = tf.keras.layers.Conv2D(embedding, (1, 1), activation=None)(x)
    return gate * enc + (1 - gate) * concept
    pass

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def main():
    pass


if __name__ == "__main__":
    main()