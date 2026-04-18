import sys
import os
import warnings
import logging
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.models.model_factory import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'v4_6')
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "base_brain_v4_5.h5")

LOOKBACK_DAYS = 60
TEST_DAYS = 500       # <--- Simulazione ultimi 500 giorni
CAPITALE_INIZIALE = 5000.0
TARGET_RISK = 0.20

TARGET_TICKERS = ['NVDA', 'META', 'AMZN', 'AAPL', 'MSFT', 'TSLA']

def get_data(ticker):
    df = yf.Ticker(ticker).history(period="3y").reset_index()
    if 'Datetime' in df.columns: df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def feature_engineering(df):
    df = df.rename(columns={'Close': 'prezzo'}).copy()
    delta = df['prezzo'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['prezzo'].shift()), abs(df['Low']-df['prezzo'].shift())], axis=1).max(axis=1)
    df['ATRr_14'] = tr.rolling(14).mean()
    df['SMA_200'] = df['prezzo'].rolling(200).mean()
    df['Dist_SMA200'] = (df['prezzo'] - df['SMA_200']) / (df['SMA_200'] + 1e-9)
    df['ret'] = df['prezzo'].pct_change().fillna(0)
    df['vol_ret'] = df['Volume'].pct_change().fillna(0)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def run_backtest():
    print(f"🚀 AVVIO BACKTEST V4.6 (Risk Manager) - Ultimi {TEST_DAYS} Giorni")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: Modello non trovato in {MODEL_PATH}")
        return

    risultati = {}
    
    for ticker in TARGET_TICKERS:
        print(f"\\nAnalisi {ticker}...")
        df = feature_engineering(get_data(ticker))
        if len(df) < LOOKBACK_DAYS + TEST_DAYS: continue

        feature_cols = ['ret', 'vol_ret', 'RSI_14', 'ATRr_14', 'Dist_SMA200']
        scaler = StandardScaler()
        
        tf.keras.backend.clear_session()
        model = get_model("4.6", MODEL_PATH, input_shape=(LOOKBACK_DAYS, len(feature_cols)))

        capitale = CAPITALE_INIZIALE
        pos = 0
        entry_price = 0.0
        
        storia_capitale = []
        df_test = df.iloc[-TEST_DAYS - LOOKBACK_DAYS:].copy()
        
        for i in tqdm(range(LOOKBACK_DAYS, len(df_test)), desc=f"Sim {ticker}"):
            finestra = df_test.iloc[i-LOOKBACK_DAYS:i]
            dati_oggi = df_test.iloc[i]
            
            prezzo_oggi = dati_oggi['prezzo']
            atr = max(dati_oggi['ATRr_14'], 1e-9)
            trend_rialzista = dati_oggi['Dist_SMA200'] > 0
            
            feat_scaled = scaler.fit_transform(finestra[feature_cols].values)
            X_pred = np.array([feat_scaled], dtype=np.float32)
            
            p_ai = model.predict(X_pred, verbose=0)[0][0]
            delta_p = p_ai - 0.50
            
            # Logica Diretta (Entry / Exit al close)
            if delta_p > 0.05: 
                nuova_pos = 1
            elif delta_p < -0.05: 
                nuova_pos = -1
            else: 
                nuova_pos = 0

            # Valuta rendimento se cambia posizione
            if nuova_pos != pos and pos != 0:
                rend = ((prezzo_oggi - entry_price) / entry_price) * (1 if pos == 1 else -1)
                capitale *= (1 + rend)
                pos = 0
                
            # Apertura nuova posizione
            if nuova_pos != 0 and pos == 0:
                pos = nuova_pos
                entry_price = prezzo_oggi
                
            # Calcolo Mark-To-Market per il grafico
            cap_fluttuante = capitale
            if pos != 0:
                rend_mkt = ((prezzo_oggi - entry_price) / entry_price) * (1 if pos == 1 else -1)
                cap_fluttuante = capitale * (1 + rend_mkt)
                
            storia_capitale.append(cap_fluttuante)

        bh_ret = ((df_test.iloc[-1]['prezzo'] - df_test.iloc[LOOKBACK_DAYS]['prezzo']) / df_test.iloc[LOOKBACK_DAYS]['prezzo']) * 100
        ai_ret = ((capitale - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100
        
        risultati[ticker] = {'AI_Return': ai_ret, 'BH_Return': bh_ret, 'Equity': storia_capitale, 'Dates': df_test['Date'].iloc[LOOKBACK_DAYS:].values}

    # --- REPORT ---
    txt = f"=== REPORT V4.6 ({TEST_DAYS} GG) ===\\n"
    for t, r in risultati.items(): txt += f"{t}: V4.6 = {r['AI_Return']:+.2f}% | B&H = {r['BH_Return']:+.2f}%\\n"
    
    print(txt)
    with open(os.path.join(REPORT_DIR, "backtest_summary.txt"), "w") as f: f.write(txt)
        
    plt.figure(figsize=(14, 7))
    for t, r in risultati.items(): plt.plot(r['Dates'], r['Equity'], label=f"{t} (V4.6)")
    plt.title(f"Equity Curve V4.6 - {TEST_DAYS} Giorni")
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "backtest_equity.png"))
    print("✅ Simulazione 4.6 terminata!")

if __name__ == "__main__":
    run_backtest()