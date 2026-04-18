import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, MaxPooling1D, Dense, Dropout, LayerNormalization
import numpy as np

class PositionalEncoding(Layer):
    """
    Aggiunge Positional Encoding temporale e standard (sin/cos) agli input.
    """
    def __init__(self, max_steps, max_dims, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims

    def build(self, input_shape):
        pos = np.arange(self.max_steps)[:, np.newaxis]
        i = np.arange(self.max_dims)[np.newaxis, :]
        d_model = self.max_dims
        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        
        pe = np.zeros(angle_rads.shape)
        pe[:, 0::2] = sines
        pe[:, 1::2] = cosines
        
        self.pe = tf.Variable(pe, dtype=tf.float32, trainable=False, name="pe_matrix")
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[tf.newaxis, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_steps": self.max_steps,
            "max_dims": self.max_dims
        })
        return config


class ProbSparseAttention(Layer):
    """
    Implementazione approssimata di ProbSparse Self-Attention (Informer).
    Calcola l'attention solo per le Top-U queries selezionate analizzando
    un sample casuale di K keys, ottimizzando teoricamente $O(L \log L)$.
    """
    def __init__(self, num_heads, key_dim, c_factor=5, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.c_factor = c_factor
        
    def build(self, input_shape):
        self.q_proj = Dense(self.num_heads * self.key_dim)
        self.k_proj = Dense(self.num_heads * self.key_dim)
        self.v_proj = Dense(self.num_heads * self.key_dim)
        self.out_proj = Dense(self.num_heads * self.key_dim)
        super().build(input_shape)

    def call(self, query, value, key=None, training=False):
        if key is None:
            key = value

        batch_size = tf.shape(query)[0]
        q_seq_len = tf.shape(query)[1]
        k_seq_len = tf.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = tf.reshape(q, (batch_size, q_seq_len, self.num_heads, self.key_dim))
        k = tf.reshape(k, (batch_size, k_seq_len, self.num_heads, self.key_dim))
        v = tf.reshape(v, (batch_size, k_seq_len, self.num_heads, self.key_dim))

        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        # Approx ProbSparse
        u_k = tf.cast(self.c_factor * tf.math.log(tf.cast(k_seq_len, tf.float32) + 1e-5), tf.int32)
        u_k = tf.clip_by_value(u_k, 1, k_seq_len)

        u_q = tf.cast(self.c_factor * tf.math.log(tf.cast(q_seq_len, tf.float32) + 1e-5), tf.int32)
        u_q = tf.clip_by_value(u_q, 1, q_seq_len)
        
        idx = tf.random.shuffle(tf.range(k_seq_len))[:u_k]
        k_sample = tf.gather(k, idx, axis=2)
        
        # Misura di sparsità per le queries
        scores_sample = tf.matmul(q, k_sample, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        sparsity_scores = tf.reduce_max(scores_sample, axis=-1) - tf.reduce_mean(scores_sample, axis=-1)
        
        # Keras tf.math.top_k ha problemi se k > dim, clip di sicurezza
        actual_u_q = tf.minimum(u_q, tf.shape(sparsity_scores)[-1])
        top_u_q, _ = tf.math.top_k(sparsity_scores, k=actual_u_q)
        
        threshold = tf.reduce_min(top_u_q, axis=-1, keepdims=True)
        is_active = sparsity_scores >= threshold
        is_active_mask = tf.expand_dims(is_active, -1)
        
        # Attention classica
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attn_weights = tf.nn.softmax(scores, axis=-1)
        context_active = tf.matmul(attn_weights, v)
        
        mean_v = tf.reduce_mean(v, axis=2, keepdims=True)
        
        context = tf.where(is_active_mask, context_active, tf.repeat(mean_v, q_seq_len, axis=2))
        
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, q_seq_len, self.num_heads * self.key_dim))
        
        return self.out_proj(context)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim, "c_factor": self.c_factor})
        return config


class DistillationLayer(Layer):
    """
    Self-Attention Distilling dell'Informer. Dimezza la dimensione temporale
    del tensore in input tramite convoluzione 1D e max pooling.
    """
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv1D(filters=filters, kernel_size=3, padding='same', activation='elu')
        self.pool = MaxPooling1D(pool_size=3, strides=2, padding='same')

    def call(self, inputs):
        x = self.conv(inputs)
        return self.pool(x)

    def get_config(self):
        config = super().get_config()
        # Per permettere il salvataggio del modello custom
        config.update({"filters": self.conv.filters})
        return config

CUSTOM_OBJECTS = {
    'PositionalEncoding': PositionalEncoding,
    'ProbSparseAttention': ProbSparseAttention,
    'DistillationLayer': DistillationLayer
}
