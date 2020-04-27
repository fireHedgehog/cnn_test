import tensorflow.compat.v1 as tf

print (tf.__version__)
tf.config.list_physical_devices('GPU')
# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# gpu_options = tf.GPUOptions(allow_growth=True)
# session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
