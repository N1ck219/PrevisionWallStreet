import os
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import ccxt

from strategies.base_strategy import BaseStrategy
from core.config import MODELS_DIR, DB_MARKET, TARGET_TICKERS_CRIPTO
from core.models.model_factory import get_model
from core.data.features import FeatureEngine

LOOKBACK_DAYS = 60
TARGET_RISK = 0.20
MARGINE = 0.06

load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
SECRET = os.getenv("BINANCE_SECRET", "").strip()

exchange = None
if API_KEY and SECRET:
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY, 'secret': SECRET,
            'enableRateLimit': True, 'options': {'defaultType': 'future'}
        })
    except: pass


class StrategyCryptoV17(BaseStrategy):
    bot_name = "CRIPTO LIVE V1.7 (Transfer Learning)"
    db_trades_path = ""  # Non usa DB trade dedicato
    model_path = os.path.join(MODELS_DIR, "crypto_base_master_v1_4.h5")
    use_macro = False  # Crypto non usa dati macro

    FEATURE_COLS = ['ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA50', 'OBV_ret']

    def setup_trades_db(self):
        pass  # Nessun DB trade per la crypto

    def execute(self):
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
                df_raw = self.get_market_data(ticker)
                if df_raw.empty: continue

                df_raw.rename(columns={'Close': 'prezzo'}, inplace=True)
                df_feat = FeatureEngine.process_crypto_features(df_raw)
                if len(df_feat) < LOOKBACK_DAYS + 10: continue

                feat_scaled = StandardScaler().fit_transform(df_feat[self.FEATURE_COLS].values)
                X_all = np.array([feat_scaled[i-LOOKBACK_DAYS:i] for i in range(LOOKBACK_DAYS, len(feat_scaled) + 1)])
                X_train, X_predict = X_all[:-1], X_all[-1:]

                # --- FASE 1: Zero-Shot ---
                tf.keras.backend.clear_session()
                model_zs = get_model("crypto_1.7", self.model_path, input_shape=(LOOKBACK_DAYS, len(self.FEATURE_COLS)))
                p_ai_zs = model_zs.predict(X_predict, verbose=0)[0][0]

                # --- FASE 2: Fine-Tuning ---
                y_train = (df_feat['prezzo'].shift(-5) > df_feat['prezzo']).astype(int).values[LOOKBACK_DAYS:-1]
                model_zs.get_layer("lstm_base").trainable = False
                if len(X_train) > 100:
                    model_zs.fit(X_train[-200:], np.array(y_train[-200:]), batch_size=32, epochs=3, verbose=0)
                p_ai_ft = model_zs.predict(X_predict, verbose=0)[0][0]

                # --- DECISIONE ---
                delta_p = ((p_ai_zs + p_ai_ft) / 2) - 0.50
                prezzo_attuale = df_feat['prezzo'].iloc[-1]

                if delta_p > MARGINE: azione = "🟢 LONG"
                elif delta_p < -MARGINE: azione = "🔴 SHORT"
                else: azione = "⚪ CASH"

                report_trades += f"<b>{ticker.split('-')[0]}</b>: {azione} (ZS: {p_ai_zs:.2f} | FT: {p_ai_ft:.2f})\n"

                # --- Esecuzione Ordine Binance ---
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
        self.set_report_data(balance_str=balance_str, trades_str=report_trades)


def run():
    StrategyCryptoV17().run()

if __name__ == "__main__":
    run()