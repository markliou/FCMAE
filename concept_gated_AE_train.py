import tensorflow as tf 
import numpy as np
import PIL
import PIL.Image
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

import concept_gated_conv

# loading the dataset
def bean_img_iter(bs = 32):
    img_size = (500, 500)
    
    dataset = tfds.load("beans", split='train', shuffle_files=True)
    dataset = dataset.batch(bs, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return iter(dataset)



dsIter = bean_img_iter(32)
cgae = concept_gated_conv.concept_gated_conv_ae()
opt = tf.keras.optimizers.AdamW(learning_rate=1e-4, global_clipnorm=1)
opt_steps = 5000000

for step in range(opt_steps):
    def ae_loss():
        ds = tf.image.resize(next(dsIter)['image'], (256, 256)) 
        
        # augmentation
        ds = tf.keras.layers.RandomFlip("horizontal_and_vertical")(ds)
        ds = tf.keras.layers.RandomRotation(0.2)(ds)
        ds = tf.keras.layers.RandomBrightness(factor=0.2)(ds)
        ds = tf.keras.layers.RandomContrast(.2)(ds)
        # ds = tf.keras.layers.RandomTranslation((.2), (.2))(ds)
        ds = tf.keras.layers.RandomZoom((.6), (.6))(ds)
        
        masked_ds = concept_gated_conv.masking_img(ds)
        ds = (ds - 128.) / 256.
        masked_ds = ((ds - 128.) / 256. - 128.) / 256.
        reconstructed_img = cgae(masked_ds)
        ae_loss = tf.keras.losses.MeanSquaredError()(reconstructed_img, ds)
        total_loss = ae_loss + tf.reduce_sum(cgae.losses)
        
        # output
        print(total_loss)
        if step % 100 == 0:
            img = PIL.Image.fromarray(tf.cast(reconstructed_img[0], dtype=tf.uint8).numpy())
            img.save('current.png')
        return total_loss
    
    opt.minimize(loss=ae_loss, var_list=cgae.trainable_weights)
