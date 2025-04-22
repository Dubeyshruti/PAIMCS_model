import os
import numpy as np
import torch
import sentencepiece as spm
import tensorflow as tf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
import sys
sys.path.append("..")
from model.paimcsLM import paimcsLm

# ──────────────────────────────────────────────────────────────────────────────
# 1. Extract teacher logits with BitsAndBytes (4‑bit NF4) on GPU
# ──────────────────────────────────────────────────────────────────────────────
def extract_teacher_logits(
    corpus_path: str,
    teacher_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    max_length: int = 678,
    save_logits: str = "teacher_logits.dat",
    save_ids: str = "input_ids.dat"
):
    if os.path.exists(save_logits) and os.path.exists(save_ids):
        return save_logits, save_ids
    # 1) Load teacher + tokenizer
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, quantization_config=bnb, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(teacher_name, use_fast=False)
    tok.pad_token = tok.eos_token

    # 2) Count lines to pre-allocate memmap
    with open(corpus_path, "r", encoding="utf-8") as f:
        N = sum(1 for _ in f if _.strip())

    V = teacher.config.vocab_size

    # 3) Create on‑disk memmaps
    logits_mm = np.memmap(save_logits,
                          dtype=np.float32,
                          mode="w+",
                          shape=(N, max_length, V))
    ids_mm    = np.memmap(save_ids,
                          dtype=np.int32,
                          mode="w+",
                          shape=(N, max_length))

    # 4) Stream again, fill memmaps
    idx = 0
    with torch.no_grad(), open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=N):
            line = line.strip()
            if not line:
                continue
            enc = tok(
                line,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            input_ids = enc.input_ids.to(teacher.device)  # shape (1, L)
            out = teacher(input_ids)

            # Move to cpu + numpy
            l = out.logits[0].cpu().float().numpy()   # (L, V)
            i = input_ids[0].cpu().numpy()            # (L,)

            logits_mm[idx] = l
            ids_mm[idx]    = i
            idx += 1

    # 5) Flush to disk
    logits_mm.flush()
    ids_mm.flush()

    print(f"Saved teacher logits → {save_logits}  and  ids → {save_ids}  ({N} examples)")
    return save_logits, save_ids

# ──────────────────────────────────────────────────────────────────────────────
# 2. Build teacher→student conversion matrix C ∈ ℝᵀˣˢ
# ──────────────────────────────────────────────────────────────────────────────
def build_conversion_matrix(
    teacher_name: str,
    sp_model: str,
    save_path: str = "teacher_to_student.npy"
):
    if os.path.exists(save_path):
        return save_path
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name)
    student_sp  = spm.SentencePieceProcessor()
    student_sp.Load(sp_model)

    T = teacher_tok.vocab_size
    S = student_sp.GetPieceSize()

    C = np.zeros((T, S), dtype=np.float32)
    for t_id in range(T):
        piece = teacher_tok.convert_ids_to_tokens(t_id)
        s_ids = student_sp.EncodeAsIds(piece)
        if len(s_ids) == 1:
            C[t_id, s_ids[0]] = 1.0
        else:
            for sid in s_ids:
                C[t_id, sid] += 1.0 / len(s_ids)

    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    C = C / row_sums

    np.save(save_path, C)
    print(f"Saved conversion matrix → {save_path} (shape {C.shape})")
    return save_path

# ──────────────────────────────────────────────────────────────────────────────
# 3. Prepare TF Datasets (train/val split, padded batches)
# ──────────────────────────────────────────────────────────────────────────────
def make_datasets(
    logits_path, ids_path, pad_token_id, teacher_vocab_size,
    batch_size=8, val_frac=0.1
):
    # get N without loading data
    logits_mm = np.memmap(logits_path, mode="r", dtype=np.float32)
    N = logits_mm.size // (678 * teacher_vocab_size)

    # create an index list and split
    idx = np.arange(N)
    np.random.shuffle(idx)
    cut = int(N * (1 - val_frac))
    train_idx, val_idx = idx[:cut], idx[cut:]

    def gen(idxs):
        logits_mm_ = np.memmap(logits_path, dtype=np.float32, mode="r")
        ids_mm_    = np.memmap(ids_path,    dtype=np.int32,   mode="r")
        for i in idxs:
            start_logits = i * 678 * teacher_vocab_size
            start_ids     = i * 678
            # slice out one example
            logits = logits_mm_[start_logits : start_logits + 678 * teacher_vocab_size]
            logits = logits.reshape(678, teacher_vocab_size)
            ids    = ids_mm_[start_ids : start_ids + 678]
            yield ids, logits

    def ds_from_indices(idxs):
        ds = tf.data.Dataset.from_generator(
            lambda: gen(idxs),
            output_signature=(
                tf.TensorSpec(shape=(678,), dtype=tf.int32),
                tf.TensorSpec(shape=(678, teacher_vocab_size), dtype=tf.float32),
            )
        )
        ds = ds.map(lambda ids, logits: {
            "input_ids":      ids,
            "labels":         ids,
            "teacher_logits": logits,
        }, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.padded_batch(
            batch_size,
            padded_shapes={
                "input_ids":      [None],
                "labels":         [None],
                "teacher_logits": [None, teacher_vocab_size],
            },
            padding_values={
                "input_ids":      pad_token_id,
                "labels":         -100,
                "teacher_logits": 0.0,
            }
        ).prefetch(tf.data.AUTOTUNE)

    return ds_from_indices(train_idx), ds_from_indices(val_idx)

# -----------------------------------------------------
# 4. DistilPaicsLM
# -----------------------------------------------------
class DistilPaicsLM(tf.keras.Model):
    def __init__(self, student, temperature=2.0, alpha=0.0, conv_matrix=None):
        super().__init__()
        self.student = student
        self.T = temperature
        self.alpha = alpha
        self.loss_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_kl = tf.keras.losses.KLDivergence()
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        # Store the conversion matrix
        if conv_matrix is None:
            raise ValueError("Must pass a conversion matrix tensor")
        self.C_tf = conv_matrix

    def call(self, inputs, training=False):
        """
        Forward pass for Keras build & predict. 
        inputs may be either:
          - a dict with key "input_ids" (from your dataset), or
          - a bare tensor of token IDs.
        """
        if isinstance(inputs, dict):
            x = inputs["input_ids"]
        else:
            x = inputs
        # delegate to your paimcsLm student
        return self.student(x, training=training)
    
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
    
    def train_step(self, batch):
        x = batch["input_ids"]
        y = batch["labels"]
        t_logits = batch["teacher_logits"]  # float32

        with tf.GradientTape() as tape:
            # Student forward pass
            s_logits = self.student(x, training=True)  # shape [B, S]

            # Hard loss (CrossEntropy on last token)
            hard = self.loss_ce(y[:, -1], s_logits)

            # Map teacher logits to student space
            t_last = t_logits[:, -1, :]  # Get the last token's teacher logits [B, T]
            mapped = tf.matmul(t_last, self.C_tf)  # Map using the conversion matrix [B, S]

            # Soft loss (KL divergence with temperature)
            T = self.T
            p_student = tf.nn.log_softmax(s_logits / T, axis=-1)  # [B, S]
            p_teacher = tf.nn.softmax(mapped / T, axis=-1)  # [B, S]
            soft = self.loss_kl(p_teacher, p_student) * (T * T)

            # Combined loss
            loss = self.alpha * hard + (1.0 - self.alpha) * soft

        # Backpropagation
        grads = tape.gradient(loss, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self.metric_loss.update_state(loss)
        return {"loss": self.metric_loss.result()}

    def test_step(self, batch):
        x = batch["input_ids"]
        y = batch["labels"]
        s_logits = self.student(x, training=False)
        loss = self.loss_ce(y[:, -1], s_logits)
        return {"loss": loss}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Distributed distillation training + saving
# ──────────────────────────────────────────────────────────────────────────────
def train_and_save(sp_model: str, teacher_name: str, ds_train, ds_val, conv_matrix_path: str, save_dir: str='../../paimcslm'):
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)
    pad_token_id = sp.PieceToId("<pad>")

    C = np.load(conv_matrix_path)
    C_tf = tf.constant(C, dtype=tf.float32)

    strategy = tf.distribute.MirroredStrategy()
    print("Replicas:", strategy.num_replicas_in_sync)

    with strategy.scope():
        student = paimcsLm(
            vocab_size=sp.GetPieceSize(),
            max_seq_len=678,
            embedding_dim=360,
            token_input_dim=339,
            groups=3,
            num_features_per_scale=120,
            gamma_list=[99e-5, 0.099, 9.9],
            num_heads=3,
            num_random_features=87,
            num_layers=23,
            dropout_rate=0.099,
        )
        distiller = DistilPaicsLM(student, temperature=2.0, alpha=0.0, conv_matrix=C_tf)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(5e-5),
        )

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    distiller.fit(ds_train, validation_data=ds_val, epochs=13, callbacks=cbs)

    os.makedirs(save_dir, exist_ok=True)
    student.save(save_dir, save_format="tf")
    print(f"Saved distilled student model → {save_dir}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_CORPUS = "../../tokenizer_train_input.txt"
    SP_MODEL     = "../../paimcs_tokenizer.model"
    TEACHER_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

    logits_path, ids_path = extract_teacher_logits(TRAIN_CORPUS, TEACHER_NAME)
    conv_matrix = build_conversion_matrix(TEACHER_NAME, SP_MODEL)

    sp = spm.SentencePieceProcessor(); sp.Load(SP_MODEL)
    ds_train, ds_val = make_datasets(
        logits_path, ids_path,
        pad_token_id=sp.PieceToId("<pad>"),
        teacher_vocab_size=AutoTokenizer.from_pretrained(TEACHER_NAME).vocab_size
    )

    train_and_save(SP_MODEL, TEACHER_NAME, ds_train, ds_val, conv_matrix)
