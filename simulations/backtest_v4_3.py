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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Importa la Fabbrica dei Modelli
from core.model_factory import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'v4_3')
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "base_brain_v4.h5")

# --- PARAMETRI BACKTEST ---
LOOKBACK_DAYS = 60
TEST_DAYS = 500       # <--- Simulazione ultimi 500 giorni
CAPITALE_INIZIALE = 10000.0
TARGET_RISK = 0.20
FEE = 0.0015

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
    print(f"🚀 AVVIO BACKTEST V4.3 - Ultimi {TEST_DAYS} Giorni")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: Modello non trovato in {MODEL_PATH}")
        return

    risultati = {}
    
    for ticker in TARGET_TICKERS:
        print(f"\\nAnalisi {ticker}...")
        df = feature_engineering(get_data(ticker))
        if len(df) < LOOKBACK_DAYS + TEST_DAYS:
            print(f"⚠️ Dati insufficienti per {ticker}")
            continue

        feature_cols = ['ret', 'vol_ret', 'RSI_14', 'ATRr_14', 'Dist_SMA200']
        scaler = StandardScaler()
        
        # Carica Modello
        tf.keras.backend.clear_session()
        model = get_model("4.3", MODEL_PATH, input_shape=(LOOKBACK_DAYS, len(feature_cols)))

        capitale = CAPITALE_INIZIALE
        bh_capitale = CAPITALE_INIZIALE
        pos, size, high_m, low_m, entry_price = 0, 0.0, 0.0, 99999.0, 0.0
        
        storia_capitale = []
        storia_bh = []
        
        df_test = df.iloc[-TEST_DAYS - LOOKBACK_DAYS:].copy()
        
        for i in tqdm(range(LOOKBACK_DAYS, len(df_test)), desc=f"Simulazione {ticker}"):
            # Finestra dati
            finestra = df_test.iloc[i-LOOKBACK_DAYS:i]
            dati_oggi = df_test.iloc[i]
            
            prezzo_oggi = dati_oggi['prezzo']
            massimo = dati_oggi['High']
            minimo = dati_oggi['Low']
            atr = max(dati_oggi['ATRr_14'], 1e-9)
            trend_rialzista = dati_oggi['Dist_SMA200'] > 0
            
            # 1. Logica di Chiusura / Stop Loss
            if pos != 0:
                chiuso = False
                if pos == 1 and massimo > high_m: high_m = massimo
                elif pos == -1 and minimo < low_m: low_m = minimo
                
                sl_price = high_m - (4.2 * atr) if pos == 1 else low_m + (3.0 * atr)
                
                if (pos == 1 and minimo <= sl_price) or (pos == -1 and massimo >= sl_price):
                    rend = ((sl_price - entry_price) / entry_price) * size if pos == 1 else ((entry_price - sl_price) / entry_price) * size
                    capitale *= (1 + rend)
                    pos, size = 0, 0.0
                    chiuso = True
                
                if not chiuso:
                    rend_mkt = (prezzo_oggi - entry_price) / entry_price
                    rend = rend_mkt * size if pos == 1 else -rend_mkt * size
                    # Simula aggiornamento giornaliero ma non chiude

            # 2. Logica di Apertura (Previsione Rete Neurale)
            if pos == 0:
                feat_scaled = scaler.fit_transform(finestra[feature_cols].values)
                X_pred = np.array([feat_scaled], dtype=np.float32)
                p_ai = model.predict(X_pred, verbose=0)[0][0]
                delta_p = p_ai - 0.50
                
                if delta_p > 0.05:
                    pos = 1
                    size = min(1.0, TARGET_RISK / ((4.2 * atr) / prezzo_oggi))
                    entry_price, high_m = prezzo_oggi, prezzo_oggi
                    capitale *= (1 - FEE)
                elif delta_p < -0.05:
                    pos = -1
                    size = min(1.0, TARGET_RISK / ((3.0 * atr) / prezzo_oggi))
                    entry_price, low_m = prezzo_oggi, prezzo_oggi
                    capitale *= (1 - FEE)

            # Buy and Hold Naturale
            if i == LOOKBACK_DAYS: prezzo_iniziale = prezzo_oggi
            bh_capitale = CAPITALE_INIZIALE * (prezzo_oggi / prezzo_iniziale)
            
            storia_capitale.append(capitale)
            storia_bh.append(bh_capitale)
            
        risultati[ticker] = {
            'AI_Return': ((capitale - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100,
            'BH_Return': ((bh_capitale - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100,
            'Equity': storia_capitale,
            'Dates': df_test['Date'].iloc[LOOKBACK_DAYS:].values
        }

    # --- REPORT FINALE ---
    report_txt = f"=== REPORT BACKTEST V4.3 ({TEST_DAYS} GG) ===\\n"
    for t, r in risultati.items():
        report_txt += f"{t}: V4.3 = {r['AI_Return']:+.2f}% | Buy&Hold = {r['BH_Return']:+.2f}%\\n"
    
    print(report_txt)
    with open(os.path.join(REPORT_DIR, "backtest_summary.txt"), "w") as f:
        f.write(report_txt)
        
    # --- GRAFICO GLOBALE ---
    plt.figure(figsize=(14, 7))
    for t, r in risultati.items():
        plt.plot(r['Dates'], r['Equity'], label=f"{t} (V4.3)")
    plt.axhline(CAPITALE_INIZIALE, color='black', linestyle='--')
    plt.title(f"Equity Curve V4.3 - Ultimi {TEST_DAYS} Giorni")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, "backtest_equity.png"))
    print("✅ Backtest completato! Report e grafico salvati.")

if __name__ == "__main__":
    run_backtest()