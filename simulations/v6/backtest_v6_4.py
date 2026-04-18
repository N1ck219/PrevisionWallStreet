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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.models.model_factory import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'v6_4')
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "base_brain_v5_0.h5") # Split Brain Model

LOOKBACK_DAYS = 60
TEST_DAYS = 500
CAPITALE_GLOBALE = 50000.0
MAX_ALLOCATION_PER_ASSET = 0.25

TARGET_TICKERS = ['NVDA', 'META', 'AMZN', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']

def get_data(ticker):
    df = yf.Ticker(ticker).history(period="3y").reset_index()
    if 'Datetime' in df.columns: df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.date
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def feature_engineering(df_t, macro):
    df = df_t.copy()
    for label, df_m in macro.items(): df = df.merge(df_m, on='Date', how='left')
    df = df.ffill().bfill().rename(columns={'Close': 'prezzo'})
    delta = df['prezzo'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['prezzo'].shift()), abs(df['Low']-df['prezzo'].shift())], axis=1).max(axis=1)
    df['ATRr_14'] = tr.rolling(14).mean()
    df['Dist_SMA200'] = (df['prezzo'] - df['prezzo'].rolling(200).mean()) / (df['prezzo'].rolling(200).mean() + 1e-9)
    df['ret'] = df['prezzo'].pct_change().fillna(0)
    for m in macro.keys(): df[f"{m}_ret"] = df[m].pct_change().fillna(0)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def run_backtest():
    print(f"🚀 AVVIO BACKTEST V6.4 (Apex Fund) - {TEST_DAYS} Giorni")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: Modello non trovato in {MODEL_PATH}")
        return

    m_tickers = {'QQQ': 'nasdaq_close'}
    macro = {label: get_data(m)[['Date', 'Close']].rename(columns={'Close': label}) for m, label in m_tickers.items()}
    
    dati_totali = {}
    date_comuni = None

    for t in TARGET_TICKERS:
        df = feature_engineering(get_data(t), macro)
        if len(df) > LOOKBACK_DAYS + TEST_DAYS:
            dati_totali[t] = df.set_index('Date')
            if date_comuni is None: date_comuni = set(df['Date'])
            else: date_comuni = date_comuni.intersection(set(df['Date']))

    date_ordinate = sorted(list(date_comuni))[-TEST_DAYS:]
    
    tf.keras.backend.clear_session()
    # Usiamo 5 features e 1 macro per simularne il funzionamento base
    model = get_model("6.4", MODEL_PATH, shape_t=(LOOKBACK_DAYS, 4), shape_m=(LOOKBACK_DAYS, 1))

    portfolio_value = CAPITALE_GLOBALE
    cash = CAPITALE_GLOBALE
    posizioni = {t: {'pos': 0, 'qty': 0, 'entry': 0.0} for t in dati_totali.keys()}
    
    storia_equity = []
    
    for data_odierna in tqdm(date_ordinate, desc="Simulazione Portafoglio"):
        valore_azioni = 0.0
        opportunities = []
        
        for t, df in dati_totali.items():
            idx = df.index.get_loc(data_odierna)
            prezzo_oggi = df.iloc[idx]['prezzo']
            
            # M2M posizioni aperte
            p = posizioni[t]
            if p['pos'] != 0:
                valore_mercato = p['qty'] * prezzo_oggi
                valore_azioni += valore_mercato
                
                # Check Stop Loss (5%)
                rend_pct = ((prezzo_oggi - p['entry']) / p['entry']) * p['pos']
                if rend_pct <= -0.05:
                    cash += valore_mercato
                    valore_azioni -= valore_mercato
                    p['pos'], p['qty'] = 0, 0
            
            # Generazione Segnali
            finestra = df.iloc[idx-LOOKBACK_DAYS:idx]
            if len(finestra) == LOOKBACK_DAYS:
                X_t = np.array([StandardScaler().fit_transform(finestra[['ret', 'RSI_14', 'ATRr_14', 'Dist_SMA200']])], dtype=np.float32)
                X_m = np.array([StandardScaler().fit_transform(finestra[['QQQ_close_ret']])], dtype=np.float32)
                delta_p = model.predict([X_t, X_m], verbose=0)[0][0] - 0.50
                if abs(delta_p) > 0.05 and p['pos'] == 0:
                    opportunities.append({'t': t, 'delta': delta_p, 'prezzo': prezzo_oggi})

        # Aggiornamento Equity Giornaliera (prima degli acquisti per evitare double counting)
        portfolio_value = cash + valore_azioni
        storia_equity.append(portfolio_value)
        
        # Allocazione
        opportunities.sort(key=lambda x: abs(x['delta']), reverse=True)
        for opp in opportunities[:2]: # Max 2 nuovi trade al giorno
            t = opp['t']; prezzo = opp['prezzo']
            budget = min(portfolio_value * MAX_ALLOCATION_PER_ASSET, cash)
            qty = int(budget // prezzo)
            if qty > 0:
                posizioni[t] = {'pos': 1 if opp['delta'] > 0 else -1, 'qty': qty, 'entry': prezzo}
                cash -= (qty * prezzo)

    # --- REPORT ---
    plt.figure(figsize=(12, 6))
    plt.plot(date_ordinate, storia_equity, label='Portafoglio Apex Fund V6.4', color='#32CD32', linewidth=2)
    plt.axhline(CAPITALE_GLOBALE, color='red', linestyle='--')
    plt.title(f"Equity Curve Globale V6.4 - {TEST_DAYS} Giorni")
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "backtest_equity.png"))
    print(f"✅ Simulazione 6.4 terminata! Ritorno Finale: {((portfolio_value-CAPITALE_GLOBALE)/CAPITALE_GLOBALE)*100:+.2f}%")

if __name__ == "__main__":
    run_backtest()