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
    conv1 = tf.keras.layers.Conv2D(32, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv1) # down sampling 128
    conv2 = concept_conv(conv1, 32)
    conv2 = tf.keras.layers.Conv2D(64, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv2) # down sampling 64
    conv3 = concept_conv(conv2, 32)
    conv3 = tf.keras.layers.Conv2D(128, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv3) # down sampling 32
    
    latent = tf.keras.layers.Conv2D(256, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(conv3)
    latent = tf.keras.layers.Conv2D(64, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(latent)  
    latent = tf.keras.layers.Conv2D(64, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(latent)  
    latent = tf.keras.layers.Conv2D(256, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(latent)  
    
    dconv1 = concept_conv(latent, 128)
    dconv1 = tf.keras.layers.Conv2DTranspose(64, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv1) # up sampling 64
    dconv2 = concept_conv(dconv1, 32)
    dconv2 = tf.keras.layers.Conv2DTranspose(32, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv2) # up sampling 128
    dconv3 = concept_conv(dconv2, 32)
    dconv3 = tf.keras.layers.Conv2DTranspose(16, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv3) # up sampling 256
    
    out = tf.keras.layers.Conv2D(16, (1,1), activation=tf.nn.tanh)(dconv3)
    out = tf.keras.layers.Conv2D(3, (1,1), activation=None)(out)
    
    return tf.keras.Model(x, out)
    pass

def concept_gated_conv_unet_ae():
    """
    create a gated convolutional autoencoder.
    The layers of the U-net is connected with concept.
    Input: 256x256x3
    Output: 256x256x3
    """
    
    x = tf.keras.layers.Input(shape=(128, 128, 3))
    
    # pixel embedding
    enc = tf.keras.layers.Conv2D(16, (1,1), activation=mish)(x)
    
    # position embedding
    pos = tf.random.uniform(shape=(128, 128, 16), dtype=tf.float32, minval=-.05, maxval=.05)
    emb = enc + pos
    
    # concept convolution
    conv1c, concept1 = concept_conv(emb, 32, True)
    conv1 = tf.keras.layers.Conv2D(32, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv1c) # down sampling 128
    conv2c, concept2 = concept_conv(conv1, 32, True)
    conv2 = tf.keras.layers.Conv2D(64, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv2c) # down sampling 64
    conv3c, concept3 = concept_conv(conv2, 32, True)
    conv3 = tf.keras.layers.Conv2D(128, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv3c) # down sampling 32
    
    latent = tf.keras.layers.Conv2D(256, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(conv3)
    latent = tf.keras.layers.Conv2D(64, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(latent)  
    latent = tf.keras.layers.Conv2D(256, (11, 11), (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(latent)  
    
    dconv1 = concept_merged_conv(latent, 32, concept3)
    dconv1 = tf.keras.layers.Conv2DTranspose(128, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv1) # up sampling 64
    dconv2 = concept_merged_conv(dconv1, 32, concept2) + conv3c
    dconv2 = tf.keras.layers.Conv2DTranspose(64, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv2) # up sampling 128
    dconv3 = concept_merged_conv(dconv2, 32, concept1) + conv2c
    dconv3 = tf.keras.layers.Conv2DTranspose(32, (3,3), (2,2), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv3) # up sampling 256
    dconv3 += conv1c
    
    # decoding
    out = tf.keras.layers.Conv2D(16, (3,3), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(dconv3)
    out = tf.keras.layers.Conv2D(16, (3,3), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(out)
    out = tf.keras.layers.Conv2D(16, (3,3), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(out)
    out = tf.keras.layers.Conv2D(3, (1,1), padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(out)
    
    return tf.keras.Model(x, out)


def concept_conv(x, channel_no, conceptOutput = False):
    conv = tf.keras.layers.LayerNormalization(axis=-1)(x)
    conv, concept = concept_conv_block(conv, channel_no)
    conv = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(conv)
    conv = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(conv)
    if(conceptOutput):
        return conv, concept
    else:
        return conv
    
def concept_merged_conv(x, channel_no, inputConcept, conceptOutput = False):
    conv = tf.keras.layers.LayerNormalization(axis=-1)(x)
    conv, concept = concept_merging_conv_block(conv, channel_no, inputConcept)
    
    if(conceptOutput):
        return conv, concept
    else:
        return conv

def concept_conv_block(x, channel_no):
    # extract concept
    concept = concept_extract_conv(x, channel_no)
    
    # concept transoforming
    concept = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(concept)
    concept = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(concept)
    
    # concept injection
    feature = concept_injection_conv(x, concept, channel_no)
    return feature, concept

def concept_merging_conv_block(x, channel_no, inputConcept):
    # extract concept
    concept = concept_extract_conv(x, channel_no)
    
    # merging concept
    concept += inputConcept
    
    # concept transoforming
    concept = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(concept)
    concept = tf.keras.layers.Conv2D(channel_no, (1,1), kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(concept)
    
    # concept injection
    feature = concept_injection_conv(x, concept, channel_no)
    return feature, concept


def concept_extract_conv(x, channel_no):
    x_shape = x.shape
    blank = tf.zeros([1, x_shape[1], x_shape[2], channel_no], dtype=tf.float32)
    concept = concept_gated_conv(x, tf.stop_gradient(blank), 3, channel_no)
    
    return tf.math.reduce_mean(concept, axis=[1, 2], keepdims=True)

def concept_injection_conv(x, concept, channel_no):
    feature = concept_gated_conv(x, concept, 3, channel_no)
    return feature

def concept_gated_conv(x, concept, kernel_size, channel_no):
    dilation_rate = (1, 1)
    gate_1 = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), dilation_rate = dilation_rate, padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(x)
    gate_1 = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), dilation_rate = dilation_rate, padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(gate_1)
    
    dilation_rate = (2, 2)
    gate_2 = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), dilation_rate = dilation_rate, padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(x)
    gate_2 = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), dilation_rate = dilation_rate, padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(gate_2)
    
    dilation_rate = (3, 3)
    gate_3 = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), dilation_rate = dilation_rate, padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(x)
    gate_3 = tf.keras.layers.Conv2D(channel_no, (kernel_size, kernel_size), dilation_rate = dilation_rate, padding="Same", kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(gate_3)
    
    gate = tf.keras.activations.sigmoid(gate_1 + gate_2 + gate_3)

    enc = tf.keras.layers.Conv2D(channel_no, (1, 1), dilation_rate = dilation_rate, kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(x)
    enc = tf.keras.layers.Conv2D(channel_no, (1, 1), dilation_rate = dilation_rate, kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=mish)(enc)
    enc = tf.keras.layers.Conv2D(channel_no, (1, 1), dilation_rate = dilation_rate, kernel_regularizer=tf.keras.regularizers.L2(1e-3), activation=None)(enc)
    
    # augment the concept
    # concept = tf.keras.layers.Dropout(.2)(tf.zeros_like(enc) + concept)
    # concept = tf.keras.layers.GaussianDropout(.2)(tf.zeros_like(enc) + concept, training=True)
    
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
        mask = tf.zeros([imgs.shape[1], imgs.shape[2]])
        mask = tf.Variable(mask)
        patch_index = tf.random.shuffle([i for i in range(totalIndexNo)])[:candidateNo]
        
        for n in patch_index:
            x = (n // split[0]) * h_size
            y = (n % split[1]) * w_size
            mask[int(x):int(x + h_size), int(y):int(y+ w_size)].assign(1) 
        return tf.reshape(mask, [imgs.shape[1], imgs.shape[2], 1])
    
    masks = tf.map_fn(lambda x: gen_mask(), tf.ones([imgs.shape[0]]), parallel_iterations=8)
    return masks

def mask_dataset_generator(img_shape, split, masking_ratio):
    totalIndexNo = split[0] * split[1]
    candidateNo = int(totalIndexNo * (1 - masking_ratio))
    h_size = img_shape[0] // split[0]
    w_size = img_shape[1] // split[1]

    ds = tf.data.Dataset.from_generator(gen_mask,
                                    args = (img_shape, split, masking_ratio, totalIndexNo, candidateNo, h_size, w_size),
                                    output_signature = tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
                                    )
    
    return ds

def gen_mask(img_shape, split, masking_ratio, totalIndexNo, candidateNo, h_size, w_size):
    mask = tf.zeros(img_shape)
    mask = tf.Variable(mask)
    # for i in range(10):
    while(1):
        patch_index = tf.random.shuffle([i for i in range(totalIndexNo)])[:candidateNo]
        
        for n in patch_index:
            x = (n // split[0]) * h_size
            y = (n % split[1]) * w_size
            mask[int(x):int(x + h_size), int(y):int(y+ w_size)].assign(1) 
        yield tf.reshape(mask, [img_shape[0], img_shape[1], 1])
    
def mask_tensor_dataset(img_shape, split=(8,8), masking_ratio = 0.9, dataset_n = 1000):
    totalIndexNo = split[0] * split[1]
    candidateNo = int(totalIndexNo * (1 - masking_ratio))
    h_size = img_shape[0] // split[0]
    w_size = img_shape[1] // split[1]
    
    def gen_mask():
        mask = tf.zeros([img_shape[0], img_shape[1]], dtype=tf.float32)
        mask = tf.Variable(mask, dtype=tf.float32)
        patch_index = tf.random.shuffle([i for i in range(totalIndexNo)])[:candidateNo]
        
        # tf.map_fn(lambda n: mask[int((n // split[0]) * h_size):int(((n // split[0]) * h_size) + h_size), int((n % split[1]) * w_size):int(((n % split[1]) * w_size)+ w_size)].assign(1.), 
        #           patch_index, 
        #           parallel_iterations=8)
        
        for n in patch_index:
            x = (n // split[0]) * h_size
            y = (n % split[1]) * w_size
            mask[int(x):int(x + h_size), int(y):int(y+ w_size)].assign(1) 
        return tf.reshape(mask, [img_shape[0], img_shape[1], 1])
    
    masks = tf.map_fn(lambda x: gen_mask(), tf.ones([dataset_n]), parallel_iterations=8)
    masks_ds = tf.data.Dataset.from_tensor_slices(masks)
    return masks_ds

def main():
    model = concept_gated_conv_ae()
    model.summary()
    
    img_shape = (256, 256)
    split = (16, 16)
    masking_ratio = .9
    ds = mask_dataset_generator(img_shape, split, masking_ratio)
    ds = ds.batch(32, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = iter(ds)
    
    for i in range(100):
        print(ds.__next__().shape)
    
    pass


if __name__ == "__main__":
    main()
