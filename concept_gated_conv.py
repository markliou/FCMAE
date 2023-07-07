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
    enc = tf.keras.layers.Conv2D(16, (1,1), activation=mish)(x)
    
    # position embedding
    pos = tf.random.uniform(shape=(256, 256, 16), dtype=tf.float32, minval=-.05, maxval=.05)
    emb = enc + pos
    
    # concept convolution
    conv1 = concept_conv(emb, 32)
    conv1 = tf.keras.layers.Conv2D(32, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.nn.tanh)(conv1) # down sampling 128
    conv2 = concept_conv(conv1, 32)
    conv2 = tf.keras.layers.Conv2D(64, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.nn.tanh)(conv2) # down sampling 64
    conv3 = concept_conv(conv2, 32)
    conv3 = tf.keras.layers.Conv2D(128, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.nn.tanh)(conv3) # down sampling 32
    
    latent = tf.keras.layers.Conv2D(256, (1,1), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(conv3) 
    
    dconv1 = concept_conv(latent, 128)
    dconv1 = tf.keras.layers.Conv2DTranspose(64, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.nn.tanh)(dconv1) # up sampling 64
    dconv2 = concept_conv(dconv1, 32)
    dconv2 = tf.keras.layers.Conv2DTranspose(32, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.nn.tanh)(dconv2) # up sampling 128
    dconv3 = concept_conv(dconv2, 32)
    dconv3 = tf.keras.layers.Conv2DTranspose(16, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.nn.tanh)(dconv3) # up sampling 256
    
    out = tf.keras.layers.Conv2D(16, (1,1), activation=tf.nn.tanh)(dconv3)
    out = tf.keras.layers.Conv2D(3, (1,1), activation=None)(out)
    
    return tf.keras.Model(x, out)
    pass

def concept_conv(x, channel_no):
    conv = tf.keras.layers.LayerNormalization(axis=-1)(x)
    conv = concept_conv_block(conv, channel_no)
    conv = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv)
    conv = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(conv)
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
    return tf.math.reduce_mean(concept, axis=[1, 2], keepdims=True)
    pass

def concept_injection_conv(x, concept, channel_no):
    feature3 = concept_gated_conv(x, concept, 3, channel_no)
    feature5 = concept_gated_conv(x, concept, 5, channel_no)
    feature7 = concept_gated_conv(x, concept, 7, channel_no)
    return feature3 + feature5 + feature7
    pass

def concept_gated_conv(x, concept, kernel_size, channel_no):
    gate = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=tf.keras.activations.sigmoid)(x)
    enc = tf.keras.layers.Conv2D(channel_no, (1, 1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(x)
    return gate * enc + (1 - gate) * concept
    pass

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def masking_img(imgs, split=(8,8), masking_ratio = 0.9):
    totalIndexNo = split[0] * split[1]
    candidateNo = int(totalIndexNo * (1 - masking_ratio))
    h_size = imgs.shape[1] // split[0]
    w_size = imgs.shape[2] // split[1]
    
    def gen_mask():
        mask = np.zeros([imgs.shape[1], imgs.shape[2]])
        patch_index = tf.random.shuffle([i for i in range(totalIndexNo)])[:candidateNo]
        
        for n in patch_index:
            x = (n // split[0]) * h_size
            y = (n % split[1]) * w_size
            mask[int(x):int(x + h_size), int(y):int(y+ w_size)] = 1
        np.reshape(mask, [imgs.shape[1], imgs.shape[2], 1])
        return np.reshape(mask, [imgs.shape[1], imgs.shape[2], 1])
    
    masks = tf.map_fn(lambda x: gen_mask(), tf.ones([imgs.shape[0]]), parallel_iterations=30)
    return masks



def main():
    model = concept_gated_conv_ae()
    model.summary()
    
    pass


if __name__ == "__main__":
    main()
