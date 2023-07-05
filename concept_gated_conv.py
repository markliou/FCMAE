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
    pos = tf.random.uniform(shape=(256, 256, 128), dtype=tf.float32, minval=-1., maxval=1.)
    emb = enc + pos
    
    # concept convolution
    conv1 = concept_conv(emb, 64)
    conv1 = tf.keras.layers.Conv2D(128, (3,3), (2,2), padding="Same", activation=mish)(conv1) # down sampling 128
    conv2 = concept_conv(conv1, 128)
    conv2 = tf.keras.layers.Conv2D(256, (3,3), (2,2), padding="Same", activation=mish)(conv2) # down sampling 64
    conv3 = concept_conv(conv2, 256)
    conv3 = tf.keras.layers.Conv2D(512, (3,3), (2,2), padding="Same", activation=mish)(conv3) # down sampling 32
    
    latent = tf.keras.layers.Conv2D(512, (1,1), (1,1), padding="Same", activation=None)(conv3) 
    
    dconv1 = concept_conv(latent, 256)
    dconv1 = tf.keras.layers.Conv2DTranspose(128, (3,3), (2,2), padding="Same", activation=mish)(dconv1) # up sampling 64
    dconv2 = concept_conv(dconv1, 128)
    dconv2 = tf.keras.layers.Conv2DTranspose(64, (3,3), (2,2), padding="Same", activation=mish)(dconv2) # up sampling 128
    dconv3 = concept_conv(dconv2, 64)
    dconv3 = tf.keras.layers.Conv2DTranspose(32, (3,3), (2,2), padding="Same", activation=mish)(dconv3) # up sampling 256
    
    out = concept_conv(dconv3, 32)
    out = tf.keras.layers.Conv2D(32, (1,1), activation=mish)(out)
    out = tf.keras.layers.Conv2D(3, (1,1), activation=None)(out)
    
    return tf.keras.Model(x, out)
    pass

def concept_conv(x, channel_no):
    conv = tf.keras.layers.LayerNormalization()(x)
    conv = concept_conv_block(conv, channel_no)
    conv = tf.keras.layers.Conv2D(channel_no, (1,1), activation=mish)(conv)
    conv = tf.keras.layers.Conv2D(channel_no, (1,1), activation=mish)(conv)
    return conv

def concept_conv_block(x, channel_no):
    # extract concept
    concept = concept_extract_conv(x, channel_no)
    # concept injection
    feature = concept_injection_conv(x, concept, channel_no)
    return feature
    pass

def concept_extract_conv(x, channel_no):
    x_shape = x.shape
    blank = tf.zeros([1, x_shape[1], x_shape[2], channel_no])
    concept3 = concept_gated_conv(x, tf.stop_gradient(blank), 3, channel_no)
    concept5 = concept_gated_conv(x, tf.stop_gradient(blank), 5, channel_no)
    concept7 = concept_gated_conv(x, tf.stop_gradient(blank), 7, channel_no)
    concept = concept3 + concept5 + concept7
    return tf.math.reduce_sum(concept, axis=[1, 2], keepdims=True)
    pass

def concept_injection_conv(x, concept, channel_no):
    feature3 = concept_gated_conv(x, concept, 3, channel_no)
    feature5 = concept_gated_conv(x, concept, 5, channel_no)
    feature7 = concept_gated_conv(x, concept, 7, channel_no)
    return feature3 + feature5 + feature7
    pass

def concept_gated_conv(x, concept, kernel_size, channel_no):
    gate = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), padding="Same", activation=tf.keras.activations.sigmoid)(x)
    enc = tf.keras.layers.Conv2D(channel_no, (1, 1), activation=None)(x)
    return gate * enc + (1 - gate) * concept
    pass

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def main():
    model = concept_gated_conv_ae()
    model.summary()
    
    pass


if __name__ == "__main__":
    main()