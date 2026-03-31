import sys
import os
import warnings
import logging
import datetime
import time
import pandas as pd
import numpy as np
import yfinance as yf 
from sklearn.preprocessing import StandardScaler 
from dotenv import load_dotenv
import ccxt

# --- IMPORT CORE ---
from core.config import BASE_DIR, DATA_DIR, MODELS_DIR, TARGET_TICKERS_CRIPTO, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from core.model_factory import get_model
from core.notifier import TelegramNotifier
from core.features import FeatureEngine

# Disabilita log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
warnings.filterwarnings('ignore')         
logging.getLogger('tensorflow').setLevel(logging.FATAL) 
import tensorflow as tf

load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
SECRET = os.getenv("BINANCE_SECRET", "").strip()

MASTER_MODEL = os.path.join(MODELS_DIR, "crypto_base_master_v1_4.h5")

LOOKBACK_DAYS = 60
TARGET_RISK = 0.20
MARGINE = 0.06

exchange = None
if API_KEY and SECRET:
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} 
        })
    except: pass



def run():
    print(f"🚀 AVVIO CRIPTO BOT LIVE V1.7 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    if not os.path.exists(MASTER_MODEL):
        print(f"❌ Modello Master non trovato al percorso: {MASTER_MODEL}")
        return

    usdt_balance = 0.0
    if exchange:
        try:
            balance = exchange.fapiPrivateGetAccount()
            for asset in balance['assets']:
                if asset['asset'] == 'USDT': usdt_balance = float(asset['marginBalance'])
        except Exception as e: print(f"Errore lettura balance: {e}")

    report_trades = ""

    for ticker in TARGET_TICKERS_CRIPTO:
        try:
            # Scarico dati con yfinance
            df_raw = yf.Ticker(ticker).history(period="1y")
            if df_raw.empty: continue
            
            df_raw = df_raw.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Close': 'prezzo'})
            df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.tz_localize(None)
            df_feat = FeatureEngine.process_crypto_features(df_raw)
            if len(df_feat) < LOOKBACK_DAYS + 10: continue

            # Pre-processing
            feature_cols = ['ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA50', 'OBV_ret']
            feat_scaled = StandardScaler().fit_transform(df_feat[feature_cols].values)
            
            X_all = np.array([feat_scaled[i-LOOKBACK_DAYS:i] for i in range(LOOKBACK_DAYS, len(feat_scaled) + 1)])
            X_train, X_predict = X_all[:-1], X_all[-1:]
            
            # --- FASE 1: Zero-Shot (Modello Globale) ---
            tf.keras.backend.clear_session()
            model_zs = get_model("crypto_1.7", MASTER_MODEL, input_shape=(LOOKBACK_DAYS, len(feature_cols)))
            p_ai_zs = model_zs.predict(X_predict, verbose=0)[0][0]
            
            # --- FASE 2: Fine-Tuning Rapido (Specifico) ---
            y_train = (df_feat['prezzo'].shift(-5) > df_feat['prezzo']).astype(int).values[LOOKBACK_DAYS:-1]
            model_zs.get_layer("lstm_base").trainable = False 
            if len(X_train) > 100:
                model_zs.fit(X_train[-200:], np.array(y_train[-200:]), batch_size=32, epochs=3, verbose=0)
            
            p_ai_ft = model_zs.predict(X_predict, verbose=0)[0][0]
            
            # --- DECISIONE FINALE (Mix ZS + FT) ---
            delta_p = ((p_ai_zs + p_ai_ft) / 2) - 0.50
            prezzo_attuale = df_feat['prezzo'].iloc[-1]
            
            if delta_p > MARGINE: azione = "🟢 LONG"
            elif delta_p < -MARGINE: azione = "🔴 SHORT"
            else: azione = "⚪ CASH"
            
            report_trades += f"<b>{ticker.split('-')[0]}</b>: {azione} (ZS: {p_ai_zs:.2f} | FT: {p_ai_ft:.2f})\n"
            
            # --- Esecuzione Ordine Binance (Opzionale) ---
            symbol_binance = ticker.replace('-', '')
            if exchange and usdt_balance > 0 and azione != "⚪ CASH":
                try:
                    exchange.cancel_all_orders(symbol_binance)
                    posizioni = exchange.fapiPrivateGetPositionRisk()
                    for p in posizioni:
                        if p['symbol'] == symbol_binance and float(p['positionAmt']) != 0:
                            exchange.fapiPrivatePostOrder({'symbol': symbol_binance, 'side': 'SELL' if float(p['positionAmt']) > 0 else 'BUY', 'type': 'MARKET', 'quantity': abs(float(p['positionAmt']))})
                            time.sleep(1)

                    vol_size = min(0.90, TARGET_RISK / (2.5 * max(df_feat['ATRr_14'].iloc[-1], 1e-9) / prezzo_attuale))
                    amount = ((usdt_balance / len(TARGET_TICKERS_CRIPTO)) * vol_size * (min(1.0, max(0.1, abs(delta_p)/0.35)))) / prezzo_attuale
                    
                    if amount > 0:
                        if prezzo_attuale > 1000: qty_str = f"{amount:.3f}"
                        elif prezzo_attuale > 100: qty_str = f"{amount:.2f}"
                        elif prezzo_attuale > 15: qty_str = f"{amount:.1f}"
                        else: qty_str = f"{int(amount)}"
                        
                        exchange.fapiPrivatePostOrder({'symbol': symbol_binance, 'side': 'BUY' if azione == "🟢 LONG" else 'SELL', 'type': 'MARKET', 'quantity': qty_str})
                        report_trades += f"   ✅ ORDINE ESEGUITO: {'BUY' if azione == '🟢 LONG' else 'SELL'} {qty_str}\n"
                except Exception as e: report_trades += f"   ❌ ERRORE: {str(e)[:80]}...\n"
                
        except Exception as e: print(f"Errore su {ticker}: {e}")
        
    balance_str = f"{usdt_balance:,.2f} USDT" if exchange else ""
    msg_html = TelegramNotifier.build_report(
        bot_name="CRIPTO LIVE V1.7 (Transfer Learning)",
        balance_str=balance_str,
        trades_str=report_trades
    )
    
    TelegramNotifier.send_message(msg_html)
    print("✅ Esecuzione Cripto Terminata.")

if __name__ == "__main__":
    run()