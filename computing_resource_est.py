import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import concept_gated_conv

tf.disable_eager_execution()

def main():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    
    model = concept_gated_conv.concept_gated_conv_unet_ae()
    
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    
    print(model.summary())
    print("model Gflops: {}".format(flops.total_float_ops / 1E9))
    
if __name__ == "__main__":
    main()

