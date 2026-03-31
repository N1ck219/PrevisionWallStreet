import sys
import os
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.model_factory import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'crypto_v1_7')
os.makedirs(REPORT_DIR, exist_ok=True)

MASTER_MODEL = os.path.join(MODELS_DIR, "crypto_base_master_v1_4.h5")

LOOKBACK_DAYS = 60
TEST_DAYS = 500
CAPITALE_INIZIALE = 1000.0 # Per singola moneta

TARGET_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD']

def feature_engineering(df):
    df = df.copy().rename(columns={'Close': 'prezzo'})
    delta = df['prezzo'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['prezzo'].shift()), abs(df['Low']-df['prezzo'].shift())], axis=1).max(axis=1)
    df['ATRr_14'] = tr.rolling(14).mean()
    df['Dist_SMA50'] = (df['prezzo'] - df['prezzo'].rolling(50).mean()) / (df['prezzo'].rolling(50).mean() + 1e-9)
    df['ret'] = df['prezzo'].pct_change().fillna(0)
    df['Target'] = (df['prezzo'].shift(-5) > df['prezzo']).astype(int)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0).dropna()

def run_backtest():
    print(f"🚀 AVVIO BACKTEST CRIPTO V1.7 (Transfer Learning) - {TEST_DAYS} Giorni")
    
    if not os.path.exists(MASTER_MODEL):
        print(f"❌ ERRORE: Master Model non trovato in {MASTER_MODEL}")
        return

    risultati = {}
    feature_cols = ['ret', 'RSI_14', 'ATRr_14', 'Dist_SMA50']

    for ticker in TARGET_TICKERS:
        print(f"\\nAnalisi {ticker} (con Fine-Tuning periodico)...")
        df_raw = yf.Ticker(ticker).history(period="3y").reset_index()
        if 'Datetime' in df_raw.columns: df_raw.rename(columns={'Datetime': 'Date'}, inplace=True)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], utc=True).dt.tz_localize(None)
        df = feature_engineering(df_raw)
        
        if len(df) < LOOKBACK_DAYS + TEST_DAYS: continue

        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(df[feature_cols].values)
        
        tf.keras.backend.clear_session()
        # Modello Zero-Shot di base
        model_zs = get_model("crypto_1.7", MASTER_MODEL, input_shape=(LOOKBACK_DAYS, len(feature_cols)))
        model_ft = get_model("crypto_1.7", MASTER_MODEL, input_shape=(LOOKBACK_DAYS, len(feature_cols)))
        model_ft.get_layer("lstm_base").trainable = False

        capitale_zs = CAPITALE_INIZIALE
        capitale_ft = CAPITALE_INIZIALE
        
        storia_zs, storia_ft, storia_bh = [], [], []
        df_test = df.iloc[-TEST_DAYS:].copy()
        
        for i in tqdm(range(len(df) - TEST_DAYS, len(df)), desc=f"Sim {ticker}"):
            finestra_feat = feat_scaled[i-LOOKBACK_DAYS:i]
            prezzo_oggi = df.iloc[i]['prezzo']
            
            # --- Fine-Tuning mensile (ogni 30 giorni) ---
            if i % 30 == 0:
                X_train = np.array([feat_scaled[j-LOOKBACK_DAYS:j] for j in range(i-200, i)])
                y_train = df['Target'].values[i-200:i]
                model_ft.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
            
            X_pred = np.array([finestra_feat], dtype=np.float32)
            p_zs = model_zs.predict(X_pred, verbose=0)[0][0] - 0.50
            p_ft = model_ft.predict(X_pred, verbose=0)[0][0] - 0.50
            
            # Simulazione semplificata: Ritorno 1-day holding
            ret_tomorrow = df.iloc[i+1]['ret'] if i < len(df)-1 else 0
            
            if p_zs > 0.05: capitale_zs *= (1 + ret_tomorrow)
            elif p_zs < -0.05: capitale_zs *= (1 - ret_tomorrow)
            
            # Modello Combinato: Transfer Learning Mix
            p_mix = (p_zs + p_ft) / 2
            if p_mix > 0.05: capitale_ft *= (1 + ret_tomorrow)
            elif p_mix < -0.05: capitale_ft *= (1 - ret_tomorrow)

            storia_zs.append(capitale_zs)
            storia_ft.append(capitale_ft)
            
            if len(storia_bh) == 0: bh_entry = prezzo_oggi
            storia_bh.append(CAPITALE_INIZIALE * (prezzo_oggi / bh_entry))

        risultati[ticker] = {
            'ZS_Ret': ((capitale_zs - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100,
            'FT_Ret': ((capitale_ft - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100,
            'Dates': df_test['Date'].values,
            'Eq_ZS': storia_zs,
            'Eq_FT': storia_ft,
            'Eq_BH': storia_bh
        }

    # --- GRAFICO A/B TEST ---
    for t, r in risultati.items():
        plt.figure(figsize=(12, 5))
        plt.plot(r['Dates'], r['Eq_ZS'], label='Solo Zero-Shot', color='#2E8B57')
        plt.plot(r['Dates'], r['Eq_FT'], label='Fine-Tuned (V1.7)', color='#FF8C00', linewidth=2)
        plt.plot(r['Dates'], r['Eq_BH'], label='Buy&Hold', color='gray', linestyle='--')
        plt.title(f"{t}: A/B Test Transfer Learning ({TEST_DAYS} Giorni)")
        plt.legend()
        plt.savefig(os.path.join(REPORT_DIR, f"ab_test_{t}.png"))
        plt.close()
        
    print("✅ Simulazione Cripto 1.7 terminata! Grafici A/B test salvati.")

if __name__ == "__main__":
    run_backtest()