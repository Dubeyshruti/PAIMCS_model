import re
from pathlib import Path
import tensorflow as tf
from sys import path
path.append("..")
from paimcs_lm import build_model
import numpy as np

# Recursively get all .txt files in /kaggle/input
text_files = list(Path("/kaggle/input").rglob("*.txt"))
print(f"Found {len(text_files)} text files for training.")

def is_valid_line(line):
    stripped = tf.strings.strip(line)
    return tf.logical_not(tf.strings.regex_full_match(stripped, r"[\.\|\s]*"))

raw_all = tf.data.Dataset.from_tensor_slices([str(f) for f in text_files]) \
    .interleave(lambda x: tf.data.TextLineDataset(x).filter(is_valid_line),
                cycle_length=tf.data.AUTOTUNE,
                num_parallel_calls=tf.data.AUTOTUNE)

tf_local_val = raw_all.take(10000)
tf_local_train = raw_all.skip(10000)

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="../tokenize/paimcs_tokenizer.model")

def txt_to_tf(ds_txt, batch_size=128):
    def encode_fn(text):
        ids = sp.encode(text.numpy().decode(), out_type=int)
        return ids[:-1], ids[1:]
    def tf_encode(x):
        inp, lbl = tf.py_function(encode_fn, [x], [tf.int32, tf.int32])
        inp.set_shape([None]); lbl.set_shape([None])
        return inp, lbl
    return ds_txt.map(lambda x: tf_encode(x), num_parallel_calls=tf.data.AUTOTUNE) \
                 .padded_batch(batch_size, padded_shapes=([None],[None])) \
                 .prefetch(tf.data.AUTOTUNE)

train_ds = txt_to_tf(tf_local_train)
val_ds = txt_to_tf(tf_local_val)

# Shuffle and prefetch
auto = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(10000).prefetch(auto)
val_ds = val_ds.prefetch(auto)

# Learning rate: warmup + cosine decay
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, warmup_steps, total_steps):
        super().__init__()
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        warmup_lr = self.init_lr * (step / tf.cast(self.warmup_steps, tf.float32))

        progress = (step - self.warmup_steps) / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(progress, 1.0)))  # Clip progress to [0, 1]
        decayed_lr = self.init_lr * cosine_decay

        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)
        
    def get_config(self):
        return {
            'init_lr': self.init_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
        }

total_steps = 5310 * 4  # epochs * steps_per_epoch
lr_schedule = WarmUpCosine(init_lr=5.31e-4, warmup_steps=10000, total_steps=total_steps)

# Build & compile
model = build_model(vocab_size=32109, seq_len=1032, num_layers=27, channels=243, kernel_size=7, heads=3, window_size=53, mlp_dim=438)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#  Callbacks: earlystop, checkpoint, tensorboard
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('paimcs_lm.keras', save_weights_only=False, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='.'),
]

#  Train
model.fit(train_ds, validation_data=val_ds, epochs=4, steps_per_epoch=5310, validation_steps=201, callbacks=callbacks)

# Final save
model.save("../../paimcs_lm.keras")

# Convert to TFLite with QAT-aware ops
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_qat = converter.convert()
open('paimcs_lm.tflite','wb').write(tflite_qat)
