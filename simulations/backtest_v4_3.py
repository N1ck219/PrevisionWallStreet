import sys
import os
import time
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

# Importa la Fabbrica dei Modelli e il Motore delle Feature
from core.model_factory import get_model
from core.features import FeatureEngine
from core.config import MACRO_MAP

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

def get_data(ticker, period="5y", retries=3):
    """Recupera i dati storici utilizzando yfinance con logica di retry."""
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period).reset_index()
            if df.empty:
                raise ValueError(f"Dati vuoti per {ticker}")
            if 'Datetime' in df.columns: 
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 2
                print(f"⚠️ Errore {ticker}: {e}. Riprovo tra {wait}s... ({attempt+1}/{retries})")
                time.sleep(wait)
            else:
                print(f"❌ Errore definitivo per {ticker}: {e}")
                return pd.DataFrame()

def run_backtest():
    print(f"🚀 AVVIO BACKTEST V4.3 - Ultimi {TEST_DAYS} Giorni")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: Modello non trovato in {MODEL_PATH}")
        return

    # 1. Recupero Dati Macro (Necessari per le 13 feature di V4.3)
    print("📥 Recupero dati macro (QQQ, VIX, TNX, SOXX, GLD)...")
    macro_data = {}
    for t_macro, label in MACRO_MAP.items():
        df_m = get_data(t_macro)
        df_m = df_m.rename(columns={'Close': label})[['Date', label]]
        macro_data[label] = df_m

    risultati = {}
    feature_cols = ['ret', 'vol_ret', 'nasdaq_ret', 'vix_ret', 'tnx_ret', 'soxx_ret', 'gld_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA200', 'OBV_ret']
    
    for ticker in TARGET_TICKERS:
        print(f"\nAnalisi {ticker}...")
        
        # 2. Feature Engineering Integrata
        df_raw = get_data(ticker)
        df = FeatureEngine.process_stock_features(df_raw, macro_data)
        
        # Rinomina 'prezzo' in 'Close' internamente se necessario per il loop di backtest, 
        # ma FeatureEngine restituisce 'prezzo'.
        if 'prezzo' not in df.columns and 'Close' in df.columns:
            df = df.rename(columns={'Close': 'prezzo'})
            
        if len(df) < LOOKBACK_DAYS + TEST_DAYS:
            print(f"⚠️ Dati insufficienti per {ticker} (presenti: {len(df)}, richiesti: {LOOKBACK_DAYS + TEST_DAYS})")
            continue

        scaler = StandardScaler()
        
        # Carica Modello con shape corretta (60, 13)
        tf.keras.backend.clear_session()
        model = get_model("4.3", MODEL_PATH, input_shape=(LOOKBACK_DAYS, len(feature_cols)))

        capitale = CAPITALE_INIZIALE
        bh_capitale = CAPITALE_INIZIALE
        pos, size, high_m, low_m, entry_price = 0, 0.0, 0.0, 99999.0, 0.0
        
        storia_capitale = []
        storia_bh = []
        
        # Offset per il test
        df_test = df.iloc[-TEST_DAYS - LOOKBACK_DAYS:].copy()
        
        for i in tqdm(range(LOOKBACK_DAYS, len(df_test)), desc=f"Simulazione {ticker}"):
            # Finestra dati
            finestra = df_test.iloc[i-LOOKBACK_DAYS:i]
            dati_oggi = df_test.iloc[i]
            
            prezzo_oggi = dati_oggi['prezzo']
            massimo = dati_oggi['High']
            minimo = dati_oggi['Low']
            atr = max(dati_oggi['ATRr_14'], 1e-9)
            
            # 1. Logica di Chiusura / Stop Loss
            if pos != 0:
                chiuso = False
                if pos == 1 and massimo > high_m: high_m = massimo
                elif pos == -1 and minimo < low_m: low_m = minimo
                
                # SL Trailing: 4.2 ATR per Long, 3.0 ATR per Short
                sl_price = high_m - (4.2 * atr) if pos == 1 else low_m + (3.0 * atr)
                
                if (pos == 1 and minimo <= sl_price) or (pos == -1 and massimo >= sl_price):
                    rend = ((sl_price - entry_price) / entry_price) * size if pos == 1 else ((entry_price - sl_price) / entry_price) * size
                    capitale *= (1 + rend)
                    pos, size = 0, 0.0
                    chiuso = True

            # 2. Logica di Apertura (Previsione Rete Neurale)
            if pos == 0:
                # Scaling dinamico sulla finestra
                feat_scaled = scaler.fit_transform(finestra[feature_cols].values)
                X_pred = np.array([feat_scaled], dtype=np.float32)
                
                p_ai = model.predict(X_pred, verbose=0)[0][0]
                delta_p = p_ai - 0.50
                
                if delta_p > 0.05:
                    pos = 1
                    # Risk Management basato su ATR
                    size = min(1.0, TARGET_RISK / ((4.2 * atr) / prezzo_oggi))
                    entry_price, high_m = prezzo_oggi, prezzo_oggi
                    capitale *= (1 - FEE)
                elif delta_p < -0.05:
                    pos = -1
                    size = min(1.0, TARGET_RISK / ((3.0 * atr) / prezzo_oggi))
                    entry_price, low_m = prezzo_oggi, prezzo_oggi
                    capitale *= (1 - FEE)

            # Buy and Hold Naturale per confronto
            if i == LOOKBACK_DAYS: prezzo_iniziale = prezzo_oggi
            bh_capitale = CAPITALE_INIZIALE * (prezzo_oggi / prezzo_iniziale)
            
            storia_capitale.append(capitale)
            storia_bh.append(bh_capitale)
            
        risultati[ticker] = {
            'AI_Return': ((capitale - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100,
            'BH_Return': ((bh_capitale - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100,
            'Equity': storia_capitale,
            'Dates': df_test['data'].iloc[LOOKBACK_DAYS:].values
        }

    # --- REPORT FINALE ---
    if not risultati:
        print("❌ Nessun risultato generato. Controlla i dati.")
        return

    report_txt = f"=== REPORT BACKTEST V4.3 ({TEST_DAYS} GG) ===\n"
    for t, r in risultati.items():
        report_txt += f"{t}: V4.3 = {r['AI_Return']:+.2f}% | Buy&Hold = {r['BH_Return']:+.2f}%\n"
    
    print("\n" + report_txt)
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
    print(f"✅ Backtest completato! Report salvato in {REPORT_DIR}")

if __name__ == "__main__":
    run_backtest()