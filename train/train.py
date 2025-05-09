import re
from pathlib import Path
import tensorflow as tf
import numpy as np
import sentencepiece as spm
import sys
sys.path.append("..")
from paimcs_lm import LM

# ------------------- GPU Strategy Setup -------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (no GPU available)")

# ------------------- Data Pipeline -------------------
text_files = list(Path("/kaggle/input").rglob("*.txt"))
print(f"Found {len(text_files)} text files for training.")

sp = spm.SentencePieceProcessor(model_file="../tokenize/paimcs_tokenizer.model")

@tf.function
def is_valid_line(line):
    stripped = tf.strings.strip(line)
    return tf.logical_not(tf.strings.regex_full_match(stripped, r"[\.\|\s]*"))

def process_file(file_path):
    return tf.data.TextLineDataset(file_path).filter(is_valid_line)

def txt_to_tf(ds_txt, batch_size=8):
    def encode_fn(text):
        ids = sp.encode(text.numpy().decode('utf-8'), out_type=int)
        return ids[:-1], ids[1:]

    def tf_encode(x):
        inp, lbl = tf.py_function(func=encode_fn, inp=[x], Tout=[tf.int32, tf.int32])
        inp.set_shape([None])
        lbl.set_shape([None])
        return inp, lbl

    return (
        ds_txt
        .interleave(process_file, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)
        .map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda x, y: tf.size(x) > 0)
        .padded_batch(batch_size, padded_shapes=([None], [None]))
        .prefetch(tf.data.AUTOTUNE)
    )

file_paths = [str(f) for f in text_files]
split_index = int(0.1 * len(file_paths))
val_ds_raw = tf.data.Dataset.from_tensor_slices(file_paths[:split_index])
train_ds_raw = tf.data.Dataset.from_tensor_slices(file_paths[split_index:])

train_ds = txt_to_tf(train_ds_raw).shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_ds = txt_to_tf(val_ds_raw).prefetch(tf.data.AUTOTUNE)

# ------------------- Model & Optimization -------------------
with strategy.scope():
    model = LM()

    total_steps = 5310 * 4
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5.31e-4,
        first_decay_steps=total_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# ------------------- Callbacks -------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        '../../paimcs_lm_final.keras', save_best_only=True, monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

# ------------------- Training -------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4,
    callbacks=callbacks
)

# ------------------- Model Saving -------------------
model.save('../../paimcs_lm_final.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int32
converter.inference_output_type = tf.int32

tflite_model = converter.convert()
with open('paimcs_lm.tflite', 'wb') as f:
    f.write(tflite_model)

print('Training and conversion complete.')
