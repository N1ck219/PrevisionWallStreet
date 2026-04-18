import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from strategies.base_strategy import BaseStrategy
from core.config import MODELS_DIR, DB_TRADES_V56, TARGET_TICKERS_AZIONARIO, ALPACA_API_KEY_5_6, ALPACA_SECRET_KEY_5_6, MACRO_LABELS_ORDERED
from core.models.model_factory import get_model
from core.data.features import FeatureEngine

LOOKBACK_DAYS = 60
CAPITALE_INIZIALE = 5000.0
TARGET_RISK = 0.20

alpaca = None
if ALPACA_API_KEY_5_6 and ALPACA_SECRET_KEY_5_6:
    try: alpaca = TradingClient(ALPACA_API_KEY_5_6, ALPACA_SECRET_KEY_5_6, paper=True)
    except: pass


class StrategyV56(BaseStrategy):
    bot_name = "V5.6 TREND RIDER (Split Brain)"
    db_trades_path = DB_TRADES_V56
    model_path = os.path.join(MODELS_DIR, "base_brain_v5_0.h5")

    def setup_trades_db(self):
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS state_v56 (Ticker TEXT PRIMARY KEY, entry REAL, atr_entry REAL, highest REAL, lowest REAL, invested_ratio REAL, pos INTEGER, half_sold INTEGER, qty INTEGER)''')
        for t in TARGET_TICKERS_AZIONARIO:
            if not self.conn_trades.execute("SELECT count(*) FROM state_v56 WHERE Ticker=?", (t,)).fetchone()[0]:
                self.conn_trades.execute("INSERT INTO state_v56 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (t, 0.0, 0.0, 0.0, 999999.0, 0.0, 0, 0, 0))
        self.conn_trades.commit()

    def _load_state(self, ticker):
        row = self.conn_trades.execute("SELECT entry, atr_entry, highest, lowest, invested_ratio, pos, half_sold, qty FROM state_v56 WHERE Ticker=?", (ticker,)).fetchone()
        return {'entry': row[0], 'atr_entry': row[1], 'highest': row[2], 'lowest': row[3], 'invested_ratio': row[4], 'pos': row[5], 'half_sold': bool(row[6]), 'qty': row[7]}

    def _save_state(self, ticker, s):
        self.conn_trades.execute("UPDATE state_v56 SET entry=?, atr_entry=?, highest=?, lowest=?, invested_ratio=?, pos=?, half_sold=?, qty=? WHERE Ticker=?", (s['entry'], s['atr_entry'], s['highest'], s['lowest'], s['invested_ratio'], s['pos'], int(s['half_sold']), s['qty'], ticker))
        self.conn_trades.commit()

    def execute(self):
        df_base = FeatureEngine.process_stock_features(self.get_market_data("AAPL"), self.macro)
        t_feat, m_feat = FeatureEngine.extract_features(df_base, macro_labels=MACRO_LABELS_ORDERED)
        scaler_t = StandardScaler().fit(t_feat)
        scaler_m = StandardScaler().fit(m_feat)

        tf.keras.backend.clear_session()
        model = get_model("5.6", self.model_path, shape_t=(LOOKBACK_DAYS, t_feat.shape[1]), shape_m=(LOOKBACK_DAYS, m_feat.shape[1]))

        operazioni_fatte = []
        balance_str = ""
        if alpaca:
            try: balance_str = f"${float(alpaca.get_account().equity):,.2f}"
            except: pass

        for ticker in tqdm(TARGET_TICKERS_AZIONARIO, desc="🎯 Elaborazione Azioni"):
            df_target = FeatureEngine.process_stock_features(self.get_market_data(ticker), self.macro)
            if len(df_target) < LOOKBACK_DAYS + 1: continue

            prezzo_oggi = df_target['prezzo'].iloc[-1]
            atr_oggi = max(df_target['ATRr_14'].iloc[-1], 1e-9)
            trend_rialzista = df_target['Dist_SMA200'].iloc[-1] > 0

            t_feat_raw, m_feat_raw = FeatureEngine.extract_features(df_target, macro_labels=MACRO_LABELS_ORDERED)
            X_t = np.array([scaler_t.transform(t_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)
            X_m = np.array([scaler_m.transform(m_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)

            delta_p = model.predict([X_t, X_m], verbose=0)[0][0] - 0.50
            state = self._load_state(ticker)

            if state['pos'] != 0:
                rend_pct = ((prezzo_oggi - state['entry']) / state['entry']) * (1 if state['pos'] == 1 else -1)
                if rend_pct <= -0.06:
                    if alpaca:
                        try: alpaca.close_position(ticker); alpaca.cancel_orders(symbol=ticker); operazioni_fatte.append(f"🔴 <b>{ticker}</b>: CHIUSO per Stop Loss (6%)")
                        except: pass
                    state['pos'] = 0; state['invested_ratio'] = 0.0; state['qty'] = 0

            if state['pos'] == 0 and delta_p > 0.05 and trend_rialzista:
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

            self._save_state(ticker, state)

        trades_str = "\n".join(operazioni_fatte) if operazioni_fatte else ""
        self.set_report_data(balance_str=balance_str, trades_str=trades_str)


def run():
    StrategyV56().run()

if __name__ == "__main__":
    run()