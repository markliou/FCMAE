import tensorflow as tf 
import numpy as np
import PIL
import PIL.Image
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

import concept_gated_conv

mirrored_strategy = tf.distribute.MirroredStrategy()

# loading the dataset
def bean_img_iter(bs = 32):
    img_size = (500, 500)
    
    dataset = tfds.load("beans", split='train', shuffle_files=True)
    dataset = dataset.batch(bs, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    return dataset

# warmup and decay learning rate
def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # source : https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
    # Cosine decay
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate

dsIter = iter(bean_img_iter(64))
opt_steps = 5000000
lr = 1e-4
with mirrored_strategy.scope():
    cgae = concept_gated_conv.concept_gated_conv_ae()
    opt = tf.keras.optimizers.AdamW(lr, global_clipnorm=1)
    cgae.load_weights('./models/cgae')
    
def training_step(ds, step):
    ds = tf.image.resize(ds['image'], (256, 256)) 
    # augmentation
    ds = tf.keras.layers.RandomFlip("horizontal_and_vertical")(ds)
    ds = tf.keras.layers.RandomRotation(0.2)(ds)
    ds = tf.keras.layers.RandomBrightness(factor=0.2)(ds)
    ds = tf.keras.layers.RandomContrast(.2)(ds)
    # ds = tf.keras.layers.RandomTranslation((.2), (.2))(ds)
    ds = tf.keras.layers.RandomZoom((.6), (.6))(ds)
    
    ds = (tf.cast(ds, tf.float32) - 128.) / 128.
    masked_ds = concept_gated_conv.masking_img(ds) * ds
    
    def ae_loss():
        reconstructed_img = cgae(masked_ds)
        # reconstructed_img = cgae(ds)
        
        ae_loss = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.SUM)(reconstructed_img, ds)
        # ae_loss = tf.math.reduce_mean(tf.math.pow((reconstructed_img - ds), 2))
        total_loss = ae_loss + tf.reduce_sum(cgae.losses)
        
        return total_loss
    
    opt.lr = lr_warmup_cosine_decay(step, 100, opt_steps, target_lr=lr)
    opt.minimize(loss=ae_loss, var_list=cgae.trainable_weights)
    
    if step % 100 == 0:
        reconstructed_img = cgae(masked_ds)
        img_array = reconstructed_img[0].numpy()
        dsimg_array = ds[0].numpy()
        masked_ds_array = masked_ds[0].numpy()
        
        img_array = tf.cast((img_array + 1 ) * 128, tf.uint8)
        dsimg_array = tf.cast((dsimg_array + 1 ) * 128, tf.uint8)
        masked_ds_array = tf.cast((masked_ds_array + 1 ) * 128, tf.uint8)
        
        img = PIL.Image.fromarray(img_array.numpy(), None)
        dsimg = PIL.Image.fromarray(dsimg_array.numpy(), None)
        masked_ds_img = PIL.Image.fromarray(masked_ds_array.numpy(), None)
        
        img.save('current.jpg')
        dsimg.save('dscurrent.jpg')
        masked_ds_img.save('ds_mased_current.jpg')
    
    return ae_loss()

for step in range(opt_steps):
    ds = next(dsIter)
    per_replica_losses = mirrored_strategy.run(training_step, args=(ds, step))
    total_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    print("step:{} loss:{}".format(step,total_loss.numpy()))
    
    if step % 100 == 0:
        cgae.save_weights('./models/cgae')

