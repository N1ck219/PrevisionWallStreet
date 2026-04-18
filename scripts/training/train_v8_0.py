import os
import sys
import argparse
import h5py
import re
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import Callback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.config import MODELS_DIR, DATA_DIR, MODELS_DIR_V8
from core.models.model_factory import get_model

MODEL_PATH = os.path.join(MODELS_DIR, "intraday_informer_v8_0.h5")
H5_DATASET_PATH = os.path.join(DATA_DIR, "datasets", "informer_v8_0_data.h5")

class OneCycleLR(Callback):
    """
    Keras Callback per il OneCycleLR Policy.
    Accellera la convergenza aumentando il LR per il primo 30% del training, 
    per poi ridurlo applicando un annealing di tipo coseno.
    """
    def __init__(self, max_lr, total_steps, initial_step=0, div_factor=25., final_div_factor=1e4):
        super(OneCycleLR, self).__init__()
        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.step_size_up = int(total_steps * 0.3)
        self.step_size_down = total_steps - self.step_size_up
        self.curr_step = initial_step

    def on_train_batch_begin(self, batch, logs=None):
        self.curr_step += 1
        curr_step = tf.cast(self.curr_step, tf.float32)
        step_up = tf.cast(self.step_size_up, tf.float32)
        step_down = tf.cast(self.step_size_down, tf.float32)

        if self.curr_step <= self.step_size_up:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (curr_step / step_up)
        else:
            progress = (curr_step - step_up) / step_down
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + tf.cos(np.pi * tf.maximum(0., progress)))
            
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


@tf.keras.utils.register_keras_serializable(package="Custom", name="directional_loss")
def directional_loss(y_true, y_pred):
    """
    Penalizza maggiormente gli errori di direzione rispetto a quelli minimi di ampiezza.
    Se y_true * y_pred < 0 (i segni sono opposti), applica una forte penalità.
    Usa Huber loss di base per la stabilità rispetto ai picchi di rendimento.
    """
    huber = tf.keras.losses.huber(y_true, y_pred)
    
    # Penalita direzionale (moltiplicata x5 se i segni non baciano)
    dir_penalty = tf.maximum(0.0, -y_true * y_pred)
    
    return huber + 5.0 * dir_penalty


class HDF5DataGenerator(tf.keras.utils.Sequence):
    """
    Generatore efficiente per leggere da HDF5 senza saturare la RAM.
    Supporta training / validazione tramite indice di start ed end.
    """
    def __init__(self, h5_path, batch_size, idx_start, idx_end, shuffle=True):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.shuffle = shuffle
        self.n_samples = self.idx_end - self.idx_start
        self.indices = np.arange(self.idx_start, self.idx_end)
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.n_samples / float(self.batch_size)))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Le query non contigue ad HDF5 sono lente, ordiniamo gli indici prima di query.
        sorted_batch_idx = np.sort(batch_indices)
        
        with h5py.File(self.h5_path, 'r') as h5f:
            X_t_batch = h5f['X_t'][sorted_batch_idx]
            X_m_batch = h5f['X_m'][sorted_batch_idx]
            y_batch = h5f['y'][sorted_batch_idx]
            
        # Re-applicare eventuale shuffle originale dopo query ordinata
        if self.shuffle:
            rev_idx = np.argsort(np.argsort(batch_indices))
            X_t_batch = X_t_batch[rev_idx]
            X_m_batch = X_m_batch[rev_idx]
            y_batch = y_batch[rev_idx]
            
        return [X_t_batch, X_m_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def run_training(epochs=50, batch_size=256, target_lr=1e-3):
    print("\n" + "=" * 60)
    print("🧠 TRAINING INFORMER V8.0 — HDF5 GENERATOR")
    print("=" * 60)
    
    if not os.path.exists(H5_DATASET_PATH):
        print(f"❌ Errore: {H5_DATASET_PATH} non trovato. Esegui prepare_v8_data.py prima!")
        return False

    with h5py.File(H5_DATASET_PATH, 'r') as h5f:
        total_samples = len(h5f['y'])
        shape_t = h5f['X_t'].shape[1:]
        shape_m = h5f['X_m'].shape[1:]
        
    print(f"✅ HDF5 caricato! Sequenze Totali: {total_samples:,}")
    print(f"   Shape T: {shape_t} | Shape M: {shape_m}")
    
    # Validation split temporale (80/20) - o casuale visto che e' per batch chunkati
    split_idx = int(total_samples * 0.8)
    
    train_gen = HDF5DataGenerator(H5_DATASET_PATH, batch_size, 0, split_idx, shuffle=True)
    val_gen = HDF5DataGenerator(H5_DATASET_PATH, batch_size, split_idx, total_samples, shuffle=False)
    
    tf.keras.backend.clear_session()
    
    # Istanziazione Modello Informer
    model = get_model("8.0", shape_t=shape_t, shape_m=shape_m)
    
    # Compilazione
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Gestito dinamicamente dal Callback OneCycleLR
        loss=directional_loss,
        metrics=['mae', 'cosine_similarity']
    )
    
    print(f"\n📐 Architettura modello: Informer V8.0")
    model.summary()
    
    # --- RESUME LOGIC ---
    initial_epoch = 0
    checkpoint_pattern = os.path.join(MODELS_DIR_V8, "informer_v8_0_epoch_*.h5")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Ordina per numero di epoca (ordinamento naturale)
        checkpoints.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)) if re.search(r'epoch_(\d+)', x) else 0)
        latest_checkpoint = checkpoints[-1]
        match = re.search(r'epoch_(\d+)', latest_checkpoint)
        if match:
            initial_epoch = int(match.group(1))
            model.load_weights(latest_checkpoint)
            print(f"\n🔄 Modello intermedio trovato! Ripristino dall'epoca: {initial_epoch}")
    
    total_steps = len(train_gen) * epochs
    initial_step = len(train_gen) * initial_epoch
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR_V8, "informer_v8_0_epoch_{epoch:02d}.h5"),
            save_weights_only=True,
            save_best_only=False
        ),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True),
        OneCycleLR(max_lr=target_lr, total_steps=total_steps, initial_step=initial_step)
    ]
    
    print(f"\n🏋️ Training: {split_idx:,} campioni | Val: {total_samples - split_idx:,}")
    print(f"   Epochs target: {epochs} | Attuale: {initial_epoch}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    best_val_loss = min(history.history['val_loss'])
    print(f"\n✅ Training completato!")
    print(f"   Best Val Loss (Directional): {best_val_loss:.4f}")
    print(f"   Modello salvato in: {MODEL_PATH}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Training del modello Informer V8.0 (HDF5)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🖥️ GPU: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
            
    run_training(epochs=args.epochs, batch_size=args.batch_size, target_lr=args.lr)


if __name__ == "__main__":
    main()
