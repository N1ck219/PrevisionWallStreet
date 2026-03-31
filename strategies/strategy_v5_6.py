import sys
import os
import warnings
import logging
import time
import datetime
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler 
from tqdm import tqdm 

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- IMPORT CORE ---
from core.config import BASE_DIR, DATA_DIR, MODELS_DIR, DB_STOCK_V45, TARGET_TICKERS_AZIONARIO, ALPACA_API_KEY, ALPACA_SECRET_KEY, MACRO_MAP, MACRO_LABELS_ORDERED
from core.model_factory import get_model
from core.data_manager import DataManager
from core.notifier import TelegramNotifier
from core.features import FeatureEngine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
warnings.filterwarnings('ignore')         
logging.getLogger('tensorflow').setLevel(logging.FATAL) 

MODEL_NAME = os.path.join(MODELS_DIR, "base_brain_v5_0.h5") 

LOOKBACK_DAYS = 60
CAPITALE_INIZIALE = 5000.0  
TARGET_RISK = 0.20 
FEE_TRANSAZIONE = 0.0015 


alpaca = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    try: alpaca = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    except: pass



def setup_db(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS state_v56 (Ticker TEXT PRIMARY KEY, entry REAL, atr_entry REAL, highest REAL, lowest REAL, invested_ratio REAL, pos INTEGER, half_sold INTEGER, qty INTEGER)''')
    for t in TARGET_TICKERS_AZIONARIO:
        if not conn.execute("SELECT count(*) FROM state_v56 WHERE Ticker=?", (t,)).fetchone()[0]:
            conn.execute("INSERT INTO state_v56 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (t, 0.0, 0.0, 0.0, 999999.0, 0.0, 0, 0, 0))
    conn.commit()

def load_portfolio_state(conn, ticker):
    row = conn.execute("SELECT entry, atr_entry, highest, lowest, invested_ratio, pos, half_sold, qty FROM state_v56 WHERE Ticker=?", (ticker,)).fetchone()
    return {'entry': row[0], 'atr_entry': row[1], 'highest': row[2], 'lowest': row[3], 'invested_ratio': row[4], 'pos': row[5], 'half_sold': bool(row[6]), 'qty': row[7]}

def save_portfolio_state(conn, ticker, s):
    conn.execute("UPDATE state_v56 SET entry=?, atr_entry=?, highest=?, lowest=?, invested_ratio=?, pos=?, half_sold=?, qty=? WHERE Ticker=?", (s['entry'], s['atr_entry'], s['highest'], s['lowest'], s['invested_ratio'], s['pos'], int(s['half_sold']), s['qty'], ticker))
    conn.commit()

def run():
    print(f"🚀 AVVIO LIVE BOT V5.6 TREND RIDER (Split Brain) - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    import sqlite3
    conn = DataManager.setup_db(DB_STOCK_V45)
    setup_db(conn)
    
    m_map = MACRO_MAP
    macro = {label: DataManager.get_cached_market_data(m, conn)[['Date', 'Close']].rename(columns={'Close': label}) for m, label in m_map.items()}
    
    if not os.path.exists(MODEL_NAME):
        print(f"❌ ERRORE: File {MODEL_NAME} non trovato! Lancia prima l'addestramento.")
        return

    df_base = FeatureEngine.process_stock_features(DataManager.get_cached_market_data("AAPL", conn), macro)
    t_feat, m_feat = FeatureEngine.extract_features(df_base, macro_labels=MACRO_LABELS_ORDERED)
    scaler_t = StandardScaler().fit(t_feat)
    scaler_m = StandardScaler().fit(m_feat)

    # 🟢 Uso del Fabbricante per Modello Split Brain (Doppio Input)
    tf.keras.backend.clear_session()
    model = get_model("5.6", MODEL_NAME, shape_t=(LOOKBACK_DAYS, t_feat.shape[1]), shape_m=(LOOKBACK_DAYS, m_feat.shape[1]))

    operazioni_fatte = []
    balance_str = ""
    if alpaca:
        try: balance_str = f"${float(alpaca.get_account().equity):,.2f}"
        except: pass

    for ticker in tqdm(TARGET_TICKERS_AZIONARIO, desc="🎯 Elaborazione Azioni"):
        df_target = FeatureEngine.process_stock_features(DataManager.get_cached_market_data(ticker, conn), macro)
        if len(df_target) < LOOKBACK_DAYS + 1: continue
        
        prezzo_oggi = df_target['prezzo'].iloc[-1]
        atr_oggi = max(df_target['ATRr_14'].iloc[-1], 1e-9)
        trend_rialzista = df_target['Dist_SMA200'].iloc[-1] > 0
        
        t_feat_raw, m_feat_raw = FeatureEngine.extract_features(df_target, macro_labels=MACRO_LABELS_ORDERED)
        X_t = np.array([scaler_t.transform(t_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)
        X_m = np.array([scaler_m.transform(m_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)
        
        p_ai = model.predict([X_t, X_m], verbose=0)[0][0]
        delta_p = p_ai - 0.50
        
        state = load_portfolio_state(conn, ticker)
        
        if state['pos'] != 0:
            rend_pct = ((prezzo_oggi - state['entry']) / state['entry']) * (1 if state['pos'] == 1 else -1)
            stop_loss = 0.06 # Stop loss statico al 6%
            if rend_pct <= -stop_loss:
                esito, mot = "❌", "Stop Loss (6%)"
                if alpaca:
                    try:
                        alpaca.close_position(ticker); alpaca.cancel_orders(symbol=ticker)
                        operazioni_fatte.append(f"🔴 <b>{ticker}</b>: CHIUSO per {mot}")
                    except: pass
                state['pos'] = 0; state['invested_ratio'] = 0.0; state['qty'] = 0

        if state['pos'] == 0:
            if delta_p > 0.05 and trend_rialzista: 
                nuovo_pos = 1
                ratio_aggiunto = min(1.0, TARGET_RISK / ((4.2 * atr_oggi) / prezzo_oggi)) * min(1.0, abs(delta_p) / 0.35)
                investimento_da_fare = CAPITALE_INIZIALE * ratio_aggiunto
                qty_ord = int(investimento_da_fare // prezzo_oggi)
                
                if alpaca and qty_ord > 0:
                    try:
                        alpaca.submit_order(MarketOrderRequest(symbol=ticker, qty=qty_ord, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                        state['entry'] = prezzo_oggi; state['atr_entry'] = atr_oggi; state['invested_ratio'] = ratio_aggiunto
                        state['pos'] = 1; state['qty'] = qty_ord; state['highest'] = prezzo_oggi; state['half_sold'] = False
                        operazioni_fatte.append(f"🟢 <b>{ticker}</b>: BUY LONG (${int(investimento_da_fare)})")
                    except Exception as e: print(f"❌ Errore Alpaca: {e}")

        save_portfolio_state(conn, ticker, state)
        
    trades_str = ""
    if operazioni_fatte: 
        trades_str += "\n".join(operazioni_fatte)
        
    msg_html = TelegramNotifier.build_report(
        bot_name="V5.6 TREND RIDER (Split Brain)",
        balance_str=balance_str,
        trades_str=trades_str
    )
    
    TelegramNotifier.send_message(msg_html)
    conn.close()
    print("✅ V5.6 Completata.")

if __name__ == "__main__":
    run()