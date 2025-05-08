import re
from pathlib import Path
import tensorflow as tf
import numpy as np
import sentencepiece as spm
import tensorflow_model_optimization as tfmot
import sys
sys.path.append("..")
from paimcs_lm import LM

# ------------------- TPU Setup -------------------
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# ------------------- Data Pipeline -------------------
# Gather text files
text_files = list(Path("/kaggle/input").rglob("*.txt"))
print(f"Found {len(text_files)} text files for training.")

# SentencePiece tokenizer
sp = spm.SentencePieceProcessor(model_file="../tokenize/paimcs_tokenizer.model")

# Filter out empty or punctuation-only lines
@tf.function
def is_valid_line(line):
    stripped = tf.strings.strip(line)
    return tf.logical_not(tf.strings.regex_full_match(stripped, r"[\.\|\s]*"))

# Encode function for dataset
def txt_to_tf(ds_txt, batch_size=128):
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
        .map(lambda f: tf.data.TextLineDataset(f).filter(is_valid_line), num_parallel_calls=tf.data.AUTOTUNE)
        .flatten()
        .map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda x, y: tf.size(x) > 0)
        .padded_batch(batch_size, padded_shapes=([None], [None]))
        .prefetch(tf.data.AUTOTUNE)
    )

# Create raw dataset
raw = tf.data.Dataset.from_tensor_slices([str(f) for f in text_files])

# Split
val_ds_raw = raw.take(10000)
train_ds_raw = raw.skip(10000)

# Prepare tf datasets
train_ds = txt_to_tf(train_ds_raw).shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_ds = txt_to_tf(val_ds_raw).prefetch(tf.data.AUTOTUNE)

# ------------------- Model & Optimization -------------------
with strategy.scope():
    # Instantiate base model
    base_model = LM()

    # Apply pruning (structured followed by unstructured)
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=1000,
            end_step=10000,
        )
    }
    pruned_model = prune_low_magnitude(base_model, **pruning_params)

    # Apply quantization-aware training
    quantize_annotate = tfmot.quantization.keras.quantize_annotate_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    annotated_model = quantize_annotate(pruned_model)
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile with optimizer and LR schedule
    total_steps = 5310 * 4
    warmup = 10000
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5.31e-4,
        first_decay_steps=total_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
    qat_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# ------------------- Callbacks -------------------
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        '../../paimcs_lm_qat_pruned.keras', save_best_only=True, monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

# ------------------- Training -------------------
qat_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4,
    steps_per_epoch=5310,
    validation_steps=201,
    callbacks=callbacks
)

# ------------------- Model Saving -------------------
# Strip pruning wrappers for inference
final_model = tfmot.sparsity.keras.strip_pruning(qat_model)

# Save Keras model
final_model.save('../../paimcs_lm_final.keras')

# Convert to TFLite (QAT)
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Ensure QAT support
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int32
converter.inference_output_type = tf.int32

tflite_model = converter.convert()
with open('paimcs_lm.tflite', 'wb') as f:
    f.write(tflite_model)

print('Training, pruning, QAT, and conversion complete.')
