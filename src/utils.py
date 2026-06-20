import tensorflow as tf
import time

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Computes sparse categorical crossentropy ignoring pad tokens (labeled as -1).
    """
    # Create mask for tokens that are NOT padding (-1)
    mask = tf.not_equal(y_true, -1)
    
    # Replace padding index with 0 temporarily to prevent indexing errors in loss function
    y_true_masked = tf.where(mask, y_true, tf.zeros_like(y_true))
    
    # Calculate standard crossentropy loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred)
    
    # Cast mask to match loss type and zero out masked losses
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    # Return average loss over unmasked elements
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

def masked_accuracy(y_true, y_pred):
    """
    Computes token classification accuracy ignoring pad tokens (labeled as -1).
    """
    mask = tf.not_equal(y_true, -1)
    
    # Prevent out of bounds indices in target
    y_true_masked = tf.where(mask, y_true, tf.zeros_like(y_true))
    
    # Calculate correctness
    predictions = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
    correct = tf.equal(y_true_masked, predictions)
    
    # Zero out correctness for padded tokens
    correct = tf.logical_and(correct, mask)
    
    # Return average accuracy over unmasked tokens
    return tf.reduce_sum(tf.cast(correct, tf.float32)) / (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-8)

class EpochTimeCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that records the duration of each training epoch.
    """
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        self.epoch_times.append(duration)
