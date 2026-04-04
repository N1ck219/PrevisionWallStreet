import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.data_manager import DataManager
from core.features import FeatureEngine
from core.model_factory import get_model
from core.config import DB_MARKET, DB_MARKET_V7, TARGET_TICKERS_AZIONARIO, MACRO_MAP, MACRO_LABELS_ORDERED, MODELS_DIR

LOOKBACK_DAYS = 60

def load_data(limit_tickers=None):
    conn_market = DataManager.setup_db(DB_MARKET)
    conn_sentiment = DataManager.setup_db(DB_MARKET_V7)
    
    macro_data = {}
    for ticker, label in MACRO_MAP.items():
        try:
             df = pd.read_sql_query("SELECT Date, Close FROM market_cache WHERE Ticker=?", conn_market, params=(ticker,))
             if not df.empty:
                 df.rename(columns={'Close': label}, inplace=True)
                 macro_data[label] = df
        except: pass

    # Carica Sentiment GLOBALE (Macro Sentiment)
    try:
        df_global_sent = pd.read_sql_query("SELECT Date as data, Sentiment_Score, Confidence, Volatility FROM sentiment_cache WHERE Ticker='GLOBAL' ORDER BY Date ASC", conn_sentiment)
    except:
        df_global_sent = pd.DataFrame()

    X_tech, X_macro, X_sent, y_labels = [], [], [], []
    
    tickers_to_process = TARGET_TICKERS_AZIONARIO[:limit_tickers] if limit_tickers else TARGET_TICKERS_AZIONARIO
    
    for ticker in tickers_to_process:
        df_stock = pd.read_sql_query("SELECT Date, Open, High, Low, Close, Volume FROM market_cache WHERE Ticker=? ORDER BY Date ASC", conn_market, params=(ticker,))
        
        try:
            df_sent = pd.read_sql_query("SELECT Date as data, Sentiment_Score, Confidence, Volatility FROM sentiment_cache WHERE Ticker=? ORDER BY Date ASC", conn_sentiment, params=(ticker,))
        except:
            df_sent = pd.DataFrame()
            
        if len(df_stock) < LOOKBACK_DAYS + 10:
            continue
            
        # Il core del Triple Brain: uniamo features (Tecniche + Macro + Sentiment Specifico + Sentiment Globale)
        df_processed = FeatureEngine.process_v7_features(df_stock, macro_data, df_sent, df_global_sent)
        t_feat, m_feat, s_feat = FeatureEngine.extract_v7_features(df_processed, MACRO_LABELS_ORDERED)
        
        t_scaler = StandardScaler().fit(t_feat)
        m_scaler = StandardScaler().fit(m_feat)
        
        # Scaling del sentiment per neural network stability se non ci sono solo zeri
        # Essendo il sentiment già tra -1 e 1 non serve scaling aggressivo ma lo mettiamo per coerenza
        s_scaler = StandardScaler().fit(s_feat)
        
        t_feat = t_scaler.transform(t_feat)
        m_feat = m_scaler.transform(m_feat)
        s_feat = s_scaler.transform(s_feat)
        
        future_ret = df_processed['prezzo'].pct_change(5).shift(-5)
        # Etichetta: 1 se il rendimento futuro a 5 giorni è > 0%
        labels = (future_ret > 0).astype(int).values
        
        print(f"[{ticker}] Dataset generato ({len(df_processed)} campioni unici).")
        
        # Estrazione con sliding window lookback
        for i in range(LOOKBACK_DAYS, len(df_processed) - 5):
            X_tech.append(t_feat[i-LOOKBACK_DAYS:i])
            X_macro.append(m_feat[i-LOOKBACK_DAYS:i])
            X_sent.append(s_feat[i-LOOKBACK_DAYS:i])
            y_labels.append(labels[i])
            
    conn_market.close()
    conn_sentiment.close()
    
    return np.array(X_tech), np.array(X_macro), np.array(X_sent), np.array(y_labels)

def train_model(epochs=50, limit_tickers=None):
    print("🔥 CARICAMENTO DATI V7 TRIPLE BRAIN IN CORSO (Tecnica + Macro + AI Sentiment)...")
    X_t, X_m, X_s, y = load_data(limit_tickers=limit_tickers)
    
    if len(X_t) == 0:
        print("Non ci sono abbastanza dati per addestrare il modello.")
        return
        
    print(f"✅ Vettore globale generato: {len(X_t)} tensori di training.")

    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.15, shuffle=False)

    Xt_tr, Xt_te = X_t[idx_train], X_t[idx_test]
    Xm_tr, Xm_te = X_m[idx_train], X_m[idx_test]
    Xs_tr, Xs_te = X_s[idx_train], X_s[idx_test]
    y_tr, y_te = y[idx_train], y[idx_test]

    model = get_model("7.0", shape_t=(LOOKBACK_DAYS, X_t.shape[2]), shape_m=(LOOKBACK_DAYS, X_m.shape[2]), shape_s=(LOOKBACK_DAYS, X_s.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODELS_DIR, "triple_brain_v7_0.h5"), save_best_only=True)
    ]
    
    history = model.fit(
        [Xt_tr, Xm_tr, Xs_tr], y_tr,
        validation_data=([Xt_te, Xm_te, Xs_te], y_te),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks
    )
    
    loss, acc = model.evaluate([Xt_te, Xm_te, Xs_te], y_te, verbose=0)
    print(f"\n✅ MODELLO V7 TRIPLE BRAIN ADDESTRATO E SALVATO! (Accuracy = {acc:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Numero di epoche per l\'addestramento')
    parser.add_argument('--full', action='store_true', help='Usa tutti i ticker (altrimenti limitato ai primi 5 per test)')
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, limit_tickers=None if args.full else 5)
