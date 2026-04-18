import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.config import DB_MARKET, DB_MARKET_V70, DATA_DIR, TARGET_TICKERS_V80, MACRO_MAP, MACRO_LABELS_ORDERED
from core.data.data_manager import DataManager
from core.data.features import FeatureEngine

LOOKBACK = 60 # Default 5 ore a 5 minuti
HORIZON = 12  # Default 1 ora a 5 minuti

H5_DATASET_PATH = os.path.join(DATA_DIR, "datasets", "informer_v8_0_data.h5")
SCALER_PATH_T = os.path.join(DATA_DIR, "datasets", "scaler_t_v8.pkl")
SCALER_PATH_M = os.path.join(DATA_DIR, "datasets", "scaler_m_v8.pkl")


def build_sequences(tech_data, macro_data, prices, lookback=LOOKBACK, horizon=HORIZON):
    """
    Costruisce sequenze. Per il V8.0 il target e' il parametro di rendimento continuo (tanh compatibile).
    rendimento = (future_price - current_price) / current_price
    Viene poi moltiplicato per 10 e clippato tra -1 e 1 in modo che +10% o piu' = 1.0 (massima confidenza).
    """
    X_t, X_m, y = [], [], []
    for i in range(lookback, len(prices) - horizon):
        X_t.append(tech_data[i - lookback:i])
        X_m.append(macro_data[i - lookback:i])
        
        future_price = prices[i + horizon]
        current_price = prices[i]
        
        ret = (future_price - current_price) / current_price
        ret_clipped = np.clip(ret * 10, -1.0, 1.0) # Scale up e limita il range
        y.append(ret_clipped)
        
    return np.array(X_t, dtype=np.float32), np.array(X_m, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=60)
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 PREPARAZIONE DATASET V8.0: HDF5 INTRADAY CHUNKING")
    print("=" * 60)
    
    conn_v70 = DataManager.setup_db(DB_MARKET_V70)
    conn_daily = DataManager.setup_db(DB_MARKET)
    
    print("\n📥 Caricamento dati macro...")
    macro = {
        label: DataManager.get_cached_market_data(m, conn_daily, start_date="2000-01-01")[['Date', 'Close']].rename(columns={'Close': label})
        for m, label in MACRO_MAP.items()
    }
    
    print("\n🔍 Pass 1: Addestramento Incremetale Scaler M/T...")
    scaler_t = StandardScaler()
    scaler_m = StandardScaler()
    
    for ticker in tqdm(TARGET_TICKERS_V80, desc="Fitting Scalers"):
        df_raw = pd.read_sql_query(f"SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker='{ticker}' ORDER BY Datetime ASC", conn_v70)
        if len(df_raw) < 500: continue
        df_5min = DataManager.resample_to_5min(df_raw)
        df_feat = FeatureEngine.process_intraday_features(df_5min, macro)
        if len(df_feat) < 100: continue
        t_raw, m_raw = FeatureEngine.extract_intraday_features(df_feat, macro_labels=MACRO_LABELS_ORDERED)
        scaler_t.partial_fit(np.nan_to_num(t_raw, nan=0.0))
        scaler_m.partial_fit(np.nan_to_num(m_raw, nan=0.0))
        
    joblib.dump(scaler_t, SCALER_PATH_T)
    joblib.dump(scaler_m, SCALER_PATH_M)
    print(f"✅ Scalers salvati in: {DATA_DIR}")
    
    print("\n📦 Pass 2: Generazione Sequenze su File HDF5 (Ottimizzazione RAM)...")
    n_features_t = scaler_t.n_features_in_
    n_features_m = scaler_m.n_features_in_
    
    with h5py.File(H5_DATASET_PATH, "w") as h5f:
        dset_x_t = h5f.create_dataset("X_t", shape=(0, args.lookback, n_features_t), maxshape=(None, args.lookback, n_features_t), chunks=True, dtype='float32')
        dset_x_m = h5f.create_dataset("X_m", shape=(0, args.lookback, n_features_m), maxshape=(None, args.lookback, n_features_m), chunks=True, dtype='float32')
        dset_y = h5f.create_dataset("y", shape=(0,), maxshape=(None,), chunks=True, dtype='float32')
        
        current_idx = 0
        
        for ticker in tqdm(TARGET_TICKERS_V80, desc="Generazione Chunk"):
            df_raw = pd.read_sql_query(f"SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker='{ticker}' ORDER BY Datetime ASC", conn_v70)
            if len(df_raw) < (args.lookback + HORIZON) * 5 + 500: continue
            
            df_5min = DataManager.resample_to_5min(df_raw)
            if len(df_5min) < args.lookback + HORIZON + 100: continue
            
            df_feat = FeatureEngine.process_intraday_features(df_5min, macro)
            if len(df_feat) < args.lookback + HORIZON + 10: continue
            
            t_raw, m_raw = FeatureEngine.extract_intraday_features(df_feat, macro_labels=MACRO_LABELS_ORDERED)
            prices = df_feat['prezzo'].values if 'prezzo' in df_feat.columns else df_feat['Close'].values
            
            t_norm = scaler_t.transform(np.nan_to_num(t_raw, nan=0.0))
            m_norm = scaler_m.transform(np.nan_to_num(m_raw, nan=0.0))
            
            X_t, X_m, y = build_sequences(t_norm, m_norm, prices, lookback=args.lookback)
            
            if len(X_t) > 0:
                n_samples = len(X_t)
                
                # Resize dimension
                dset_x_t.resize(current_idx + n_samples, axis=0)
                dset_x_m.resize(current_idx + n_samples, axis=0)
                dset_y.resize(current_idx + n_samples, axis=0)
                
                # Write to HDF5 explicitly instead of concatenating in RAM
                dset_x_t[current_idx:current_idx + n_samples] = X_t
                dset_x_m[current_idx:current_idx + n_samples] = X_m
                dset_y[current_idx:current_idx + n_samples] = y
                
                current_idx += n_samples
                
    print(f"\n✅ Salvataggio completato! {current_idx:,} campioni estratti e scritti su HDF5.")
    
if __name__ == '__main__':
    main()
