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
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import TakeProfitRequest, StopLossRequest

# --- IMPORT CORE ---
from core.config import BASE_DIR, DATA_DIR, MODELS_DIR, DB_STOCK_V45, TARGET_TICKERS_AZIONARIO, ALPACA_API_KEY, ALPACA_SECRET_KEY, MACRO_MAP
from core.model_factory import get_model
from core.data_manager import DataManager
from core.notifier import TelegramNotifier
from core.features import FeatureEngine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
warnings.filterwarnings('ignore')         
logging.getLogger('tensorflow').setLevel(logging.FATAL) 

MODEL_NAME = os.path.join(MODELS_DIR, "base_brain_v4_5.h5")

LOOKBACK_DAYS = 60
CAPITALE_INIZIALE = 5000.0
TARGET_RISK = 0.20 
MARGINE = 0.05 

alpaca = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    try: alpaca = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    except: pass



def setup_live_database(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS portfolio_live_46 (Ticker TEXT PRIMARY KEY, Last_Position REAL, Last_Price REAL, Last_Date TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS history_live_46 (Date TEXT, Ticker TEXT, Win_Loss TEXT)''')
    for t in TARGET_TICKERS_AZIONARIO:
        if not conn.execute("SELECT count(*) FROM portfolio_live_46 WHERE Ticker=?", (t,)).fetchone()[0]:
            conn.execute("INSERT INTO portfolio_live_46 VALUES (?, ?, ?, ?)", (t, 0.0, 0.0, '2000-01-01'))
    conn.commit()

def get_win_rate_stats(conn):
    storico = conn.execute("SELECT Ticker, Win_Loss FROM history_live_46").fetchall()
    tot = len(storico)
    vinte = sum(1 for row in storico if row[1] == "✅")
    return f"{int((vinte/tot)*100)}% ({vinte} Vinte / {tot} Totali)" if tot > 0 else "N/A"

def run():
    print(f"🚀 AVVIO LIVE BOT V4.6 RISK MANAGER - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    import sqlite3
    conn = DataManager.setup_db(DB_STOCK_V45)
    setup_live_database(conn)
    
    m_map = MACRO_MAP
    macro = {label: DataManager.get_cached_market_data(m, conn)[['Date', 'Close']].rename(columns={'Close': label}) for m, label in m_map.items()}
    feature_cols = ['ret', 'vol_ret', 'nasdaq_ret', 'vix_ret', 'tnx_ret', 'soxx_ret', 'gld_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA200', 'OBV_ret']
    
    if not os.path.exists(MODEL_NAME):
        print(f"❌ ERRORE: File {MODEL_NAME} non trovato nella cartella models!")
        return

    df_base = FeatureEngine.process_stock_features(DataManager.get_cached_market_data("AAPL", conn), macro)
    global_scaler = StandardScaler().fit(df_base[feature_cols].values)

    report_domani = ""
    posizioni_alpaca = {p.symbol: p for p in alpaca.get_all_positions()} if alpaca else {}

    for ticker in tqdm(TARGET_TICKERS_AZIONARIO, desc="🎯 Analisi Azioni V4.6"):
        df_target = FeatureEngine.process_stock_features(DataManager.get_cached_market_data(ticker, conn), macro)
        if len(df_target) < LOOKBACK_DAYS + 1: continue
        
        data_oggi_db = str(df_target['data'].iloc[-1])
        feat_t = global_scaler.transform(df_target[feature_cols].values)
        X_predict = np.array([feat_t[-LOOKBACK_DAYS:]], dtype=np.float32)
        
        # 🟢 Uso del Fabbricante per Caricare il Modello
        tf.keras.backend.clear_session()
        model = get_model("4.6", MODEL_NAME, input_shape=(LOOKBACK_DAYS, len(feature_cols)))
        
        delta_p = model.predict(X_predict, verbose=0)[0][0] - 0.50
        prezzo_oggi = df_target['prezzo'].iloc[-1]
        atr_oggi = max(df_target['ATRr_14'].iloc[-1], 1e-9)
        trend_rialzista = df_target['Dist_SMA200'].iloc[-1] > 0
        
        pos_db, last_price = conn.execute("SELECT Last_Position, Last_Price FROM portfolio_live_46 WHERE Ticker=?", (ticker,)).fetchone()
        
        if delta_p > MARGINE: 
            np_pos = 1
            nuova_size = round(min(1.0, TARGET_RISK / ((4.2 * atr_oggi) / prezzo_oggi)) * min(1.0, max(0.1, abs(delta_p) / 0.35)), 2)
            if not trend_rialzista: nuova_size = min(nuova_size, 0.5)
            txt = f"🟢 LONG ({int(nuova_size*100)}%)"
            
        elif delta_p < -MARGINE: 
            np_pos = -1
            nuova_size = round(min(1.0, TARGET_RISK / ((3.0 * atr_oggi) / prezzo_oggi)) * min(1.0, max(0.1, abs(delta_p) / 0.35)), 2)
            if trend_rialzista: nuova_size = min(nuova_size, 0.5)
            txt = f"🔴 SHORT ({int(nuova_size*100)}%)"
        else: np_pos, nuova_size, txt = 0, 0.0, "⚪ CASH"

        cambio_stato = (np_pos != pos_db)
        if cambio_stato and pos_db != 0:
            esito = "✅" if (pos_db == 1 and prezzo_oggi > last_price) or (pos_db == -1 and prezzo_oggi < last_price) else "❌"
            conn.execute("INSERT INTO history_live_46 VALUES (?, ?, ?)", (data_oggi_db, ticker, esito)); conn.commit()

        if ticker in posizioni_alpaca: report_domani += f"<b>{ticker}</b>: {txt} | 💼 {posizioni_alpaca[ticker].qty} az.\n"
        else: report_domani += f"<b>{ticker}</b>: {txt} | 💼 0 az.\n"

        if alpaca:
            try:
                if np_pos == 0 or (cambio_stato and pos_db != 0):
                    try: alpaca.close_position(ticker); alpaca.cancel_orders(symbol=ticker); time.sleep(1)
                    except: pass 
                if cambio_stato and np_pos != 0:
                    qty_totale = int((CAPITALE_INIZIALE * nuova_size) // prezzo_oggi) 
                    if qty_totale > 0:
                        lato = OrderSide.BUY if np_pos == 1 else OrderSide.SELL
                        if qty_totale >= 2:
                            qty_meta, qty_resto = qty_totale // 2, qty_totale - (qty_totale // 2)
                            sl_price = round(prezzo_oggi - (4.2*atr_oggi), 2) if np_pos == 1 else round(prezzo_oggi + (3.0*atr_oggi), 2)
                            alpaca.submit_order(MarketOrderRequest(symbol=ticker, qty=qty_meta, side=lato, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, stop_loss=StopLossRequest(stop_price=sl_price), take_profit=TakeProfitRequest(limit_price=round(prezzo_oggi + (2.0*atr_oggi), 2) if np_pos == 1 else round(prezzo_oggi - (2.0*atr_oggi), 2))))
                            alpaca.submit_order(MarketOrderRequest(symbol=ticker, qty=qty_resto, side=lato, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, stop_loss=StopLossRequest(stop_price=sl_price), take_profit=TakeProfitRequest(limit_price=round(prezzo_oggi + (10.0*atr_oggi), 2) if np_pos == 1 else round(prezzo_oggi - (10.0*atr_oggi), 2))))
                        else: 
                            sl_price = round(prezzo_oggi - (4.2*atr_oggi), 2) if np_pos == 1 else round(prezzo_oggi + (3.0*atr_oggi), 2)
                            alpaca.submit_order(MarketOrderRequest(symbol=ticker, qty=1, side=lato, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, stop_loss=StopLossRequest(stop_price=sl_price), take_profit=TakeProfitRequest(limit_price=round(prezzo_oggi + (10.0*atr_oggi), 2) if np_pos == 1 else round(prezzo_oggi - (10.0*atr_oggi), 2))))
            except Exception as e: report_domani += f"   ❌ <i>Errore Alpaca: {e}</i>\n"

        if cambio_stato:
            conn.execute("UPDATE portfolio_live_46 SET Last_Position=?, Last_Price=?, Last_Date=? WHERE Ticker=?", (np_pos, prezzo_oggi if np_pos != 0 else 0.0, data_oggi_db, ticker))
            conn.commit()

    win_rate_str = get_win_rate_stats(conn)
    conn.close()

    balance_str = f"${float(alpaca.get_account().equity):,.2f}" if alpaca else ""
    
    msg_html = TelegramNotifier.build_report(
        bot_name="V4.6 RISK MANAGER LIVE",
        balance_str=balance_str,
        win_rate_str=win_rate_str,
        trades_str=report_domani
    )
    
    TelegramNotifier.send_message(msg_html)
    print("✅ V4.6 Completata.")

if __name__ == "__main__":
    run()