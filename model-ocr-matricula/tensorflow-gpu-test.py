import tensorflow as tf

# Check if TensorFlow can access a GPU
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs are available:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("No GPUs found.")

# Test a simple computation on GPU if available
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:\n", c.numpy())
except RuntimeError as e:
    print("Error during GPU computation:", e)