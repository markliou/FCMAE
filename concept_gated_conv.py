import tensorflow as tf 
import numpy as np

def concept_gated_conv_ae():
    """
    create a gated convolutional autoencoder
    Input: 500x500x3
    Output: 500x500x3
    """
    x = tf.keras.layers.Input(shape=(256, 256,3))
    # pixel embedding
    
    # position embedding
    
    
    out = None
    return tf.keras.Model(x, out)
    pass

def main():
    pass


if __name__ == "__main__":
    main()