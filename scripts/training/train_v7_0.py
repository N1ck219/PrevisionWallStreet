"""
train_v7_0.py — Addestramento del modello Split Brain V7.0.

Training diretto su dati intraday resampleati a 5 minuti.
I dati raw a 1 minuto vengono aggregati in barre a 5 min per:
    - Ridurre il rumore (bid-ask bounce, micro-fluttuazioni)
    - Aumentare il contesto temporale (LOOKBACK 60 barre = 5 ore di sessione)
    - Stabilizzare il target (HORIZON 12 barre = 1 ora)

Uso:
    python train_v7_0.py                    # Training completo
    python train_v7_0.py --epochs 100       # Numero di epoche personalizzato
    python train_v7_0.py --batch-size 512   # Batch size personalizzato
"""

import os
import sys
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core.config import (
    DB_MARKET, DB_MARKET_V70, MODELS_DIR, TARGET_TICKERS_V70,
    MACRO_MAP, MACRO_LABELS_ORDERED,
    ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
)
from core.data.data_manager import DataManager
from core.data.features import FeatureEngine
from core.models.model_factory import get_model

LOOKBACK = 60           # Finestra di lookback (60 barre a 5 min = 5 ore)
HORIZON = 12            # Orizzonte previsione (12 barre a 5 min = 1 ora)
MODEL_PATH = os.path.join(MODELS_DIR, "intraday_brain_v7_0.h5")

# ── Feature columns (devono matchare extract_intraday_features) ──
TECH_COLS = [
    'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
    'ATRr_14', 'Dist_SMA50', 'OBV_ret', 'VWAP_ratio',
    'minutes_since_open', 'session_pct'
]
MACRO_COLS_BASE = [f"{m}_ret" for m in MACRO_LABELS_ORDERED]
MACRO_COLS_EXTRA = ['is_power_hour', 'is_opening_range']


def build_sequences(tech_data, macro_data, prices, lookback=LOOKBACK, horizon=HORIZON):
    """Costruisce sequenze X_tech, X_macro, y per il training.
    
    y = 1 se il prezzo dopo `horizon` barre è superiore al prezzo corrente, 0 altrimenti.
    """
    X_t, X_m, y = [], [], []
    
    for i in range(lookback, len(prices) - horizon):
        X_t.append(tech_data[i - lookback:i])
        X_m.append(macro_data[i - lookback:i])
        
        # Target: direzione del prezzo dopo 1 ora (12 barre a 5 min)
        future_price = prices[i + horizon]
        current_price = prices[i]
        y.append(1.0 if future_price > current_price else 0.0)
    
    return np.array(X_t, dtype=np.float32), np.array(X_m, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_intraday_data(conn, macro):
    """Prepara dati intraday resampleati a 5 minuti per il training."""
    print("\n📊 Preparazione dati intraday (resampling 1min → 5min)...")
    
    all_X_t, all_X_m, all_y = [], [], []
    
    for ticker in tqdm(TARGET_TICKERS_V70, desc="Elaborazione ticker"):
        # Carica dati raw a 1 minuto dal DB
        df_raw = pd.read_sql_query(
            "SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker=? ORDER BY Datetime ASC",
            conn, params=(ticker,)
        )
        
        if len(df_raw) < (LOOKBACK + HORIZON) * 5 + 500:
            print(f"  ⚠️ {ticker}: dati insufficienti ({len(df_raw)} barre 1-min), skip")
            continue
        
        # Resample a 5 minuti
        df_5min = DataManager.resample_to_5min(df_raw)
        
        if len(df_5min) < LOOKBACK + HORIZON + 100:
            print(f"  ⚠️ {ticker}: dati 5-min insufficienti ({len(df_5min)} barre), skip")
            continue
        
        # Feature engineering sulle barre a 5 minuti
        df_feat = FeatureEngine.process_intraday_features(df_5min, macro)
        
        if len(df_feat) < LOOKBACK + HORIZON + 10:
            continue
        
        # Estrai feature separate per Split Brain
        t_raw, m_raw = FeatureEngine.extract_intraday_features(df_feat, macro_labels=MACRO_LABELS_ORDERED)
        
        prices = df_feat['prezzo'].values if 'prezzo' in df_feat.columns else df_feat['Close'].values
        
        # Normalizza
        scaler_t = StandardScaler()
        scaler_m = StandardScaler()
        t_norm = scaler_t.fit_transform(t_raw)
        m_norm = scaler_m.fit_transform(m_raw)
        
        # Costruisci sequenze
        X_t, X_m, y = build_sequences(t_norm, m_norm, prices)
        
        if len(X_t) > 0:
            # Campionamento: limit per ticker per bilanciare il dataset
            max_per_ticker = 80000
            if len(X_t) > max_per_ticker:
                idx = np.random.choice(len(X_t), max_per_ticker, replace=False)
                idx.sort()
                X_t, X_m, y = X_t[idx], X_m[idx], y[idx]
            
            all_X_t.append(X_t)
            all_X_m.append(X_m)
            all_y.append(y)
            print(f"  ✅ {ticker}: {len(X_t):,} campioni (da {len(df_5min):,} barre 5-min)")
    
    if not all_X_t:
        print("❌ Nessun dato intraday preparato!")
        print("   Esegui prima: python download_intraday_data.py")
        return None, None, None
    
    X_t_all = np.concatenate(all_X_t)
    X_m_all = np.concatenate(all_X_m)
    y_all = np.concatenate(all_y)
    
    # Shuffle
    perm = np.random.permutation(len(y_all))
    X_t_all, X_m_all, y_all = X_t_all[perm], X_m_all[perm], y_all[perm]
    
    print(f"\n✅ Dataset totale: {len(y_all):,} campioni")
    print(f"   Tech shape:  {X_t_all.shape}")
    print(f"   Macro shape: {X_m_all.shape}")
    print(f"   Distribuzione target: {y_all.mean():.2%} positivi")
    
    return X_t_all, X_m_all, y_all


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_t, X_m, y, batch_size, shuffle=True):
        self.X_t = X_t
        self.X_m = X_m
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return [self.X_t[batch_idx], self.X_m[batch_idx]], self.y[batch_idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def run_training(conn, macro, epochs=80, batch_size=512):
    """Training diretto su dati intraday a 5 minuti."""
    print("\n" + "=" * 60)
    print("🧠 TRAINING SPLIT BRAIN V7.0 — DATI INTRADAY 5 MINUTI")
    print("=" * 60)
    
    X_t, X_m, y = prepare_intraday_data(conn, macro)
    if X_t is None:
        return False
    
    n_tech = X_t.shape[2]
    n_macro = X_m.shape[2]
    
    tf.keras.backend.clear_session()
    model = get_model("7.0", shape_t=(LOOKBACK, n_tech), shape_m=(LOOKBACK, n_macro))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📐 Architettura modello:")
    model.summary()
    
    # Split train/val (usa gli ultimi dati come validation)
    split = int(len(y) * 0.85)
    X_t_train, X_t_val = X_t[:split], X_t[split:]
    X_m_train, X_m_val = X_m[:split], X_m[split:]
    y_train, y_val = y[:split], y[split:]
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True
        )
    ]
    
    print(f"\n🏋️ Training: {len(y_train):,} campioni | Val: {len(y_val):,}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | LR: 0.0005")
    print(f"   LOOKBACK: {LOOKBACK} barre (5 ore) | HORIZON: {HORIZON} barre (1 ora)")
    
    train_gen = DataGenerator(X_t_train, X_m_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = DataGenerator(X_t_val, X_m_val, y_val, batch_size=batch_size, shuffle=False)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salva pesi finali
    model.save_weights(MODEL_PATH)
    
    # Stampa risultati
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Training completato!")
    print(f"   Best Val Loss:     {best_val_loss:.4f}")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Modello salvato in: {MODEL_PATH}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Training del modello Split Brain V7.0 (5-min)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Numero di epoche (default: 80)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🧠 TRAINING MODELLO SPLIT BRAIN V7.0")
    print(f"   Timeframe: 5 minuti (resampled da 1 min)")
    print(f"   LOOKBACK:  {LOOKBACK} barre = 5 ore")
    print(f"   HORIZON:   {HORIZON} barre = 1 ora")
    print(f"   Data:      {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🖥️ GPU disponibile: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
    else:
        print("⚠️ Nessuna GPU trovata — training su CPU (più lento)")
    
    # Connessione DB intraday V7.0
    conn_v70 = DataManager.setup_db(DB_MARKET_V70)
    
    # Connessione DB daily per dati macro
    conn_daily = DataManager.setup_db(DB_MARKET)
    
    print("\n📥 Caricamento dati macro...")
    macro = {
        label: DataManager.get_cached_market_data(m, conn_daily, start_date="2000-01-01")[['Date', 'Close']].rename(columns={'Close': label})
        for m, label in MACRO_MAP.items()
    }
    print(f"   Caricati {len(macro)} indicatori macro")
    
    # Esecuzione training
    success = run_training(conn_v70, macro, epochs=args.epochs, batch_size=args.batch_size)
    
    conn_v70.close()
    conn_daily.close()
    
    if success:
        print("\n🏁 TRAINING TERMINATO CON SUCCESSO!")
    else:
        print("\n❌ TRAINING FALLITO — controlla i dati intraday")


if __name__ == "__main__":
    main()
