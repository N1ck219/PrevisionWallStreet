import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from core.base_strategy import BaseStrategy
from core.config import MODELS_DIR, DB_TRADES_V64, TARGET_TICKERS_AZIONARIO, MACRO_LABELS_ORDERED, DB_MARKET_V7
from core.model_factory import get_model
from core.features import FeatureEngine
from core.data_manager import DataManager
import pandas as pd

LOOKBACK_DAYS = 60
CAPITALE_INIZIALE = 5000.0
TARGET_RISK = 0.20

class StrategyV70(BaseStrategy):
    bot_name = "V7.0 TRIPLE BRAIN"
    # Useremo un DB nuovo per questa strategia, se vuoi ricicliamo db_trades
    db_trades_path = os.path.join(os.path.dirname(DB_TRADES_V64), "trades_v7_0.db")
    model_path = os.path.join(MODELS_DIR, "triple_brain_v7_0.h5")

    def __init__(self):
        super().__init__()
        self.conn_market_v7 = DataManager.setup_db(DB_MARKET_V7)

    def setup_trades_db(self):
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS state_v70 (Ticker TEXT PRIMARY KEY, entry REAL, atr_entry REAL, highest REAL, lowest REAL, invested_ratio REAL, pos INTEGER, half_sold INTEGER, qty INTEGER)''')
        for t in TARGET_TICKERS_AZIONARIO:
            if not self.conn_trades.execute("SELECT count(*) FROM state_v70 WHERE Ticker=?", (t,)).fetchone()[0]:
                self.conn_trades.execute("INSERT INTO state_v70 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (t, 0.0, 0.0, 0.0, 999999.0, 0.0, 0, 0, 0))
        self.conn_trades.commit()

    def _load_state(self, ticker):
        row = self.conn_trades.execute("SELECT entry, atr_entry, highest, lowest, invested_ratio, pos, half_sold, qty FROM state_v70 WHERE Ticker=?", (ticker,)).fetchone()
        return {'entry': row[0], 'atr_entry': row[1], 'highest': row[2], 'lowest': row[3], 'invested_ratio': row[4], 'pos': row[5], 'half_sold': bool(row[6]), 'qty': row[7]}

    def _save_state(self, ticker, s):
        self.conn_trades.execute("UPDATE state_v70 SET entry=?, atr_entry=?, highest=?, lowest=?, invested_ratio=?, pos=?, half_sold=?, qty=? WHERE Ticker=?", (s['entry'], s['atr_entry'], s['highest'], s['lowest'], s['invested_ratio'], s['pos'], int(s['half_sold']), s['qty'], ticker))
        self.conn_trades.commit()

    def get_sentiment_data(self, ticker):
        """Recupera i dati di sentiment dal DB V7."""
        try:
             df_sent = pd.read_sql_query("SELECT Date as data, Sentiment_Score, Confidence, Volatility FROM sentiment_cache WHERE Ticker=?", self.conn_market_v7, params=(ticker,))
             return df_sent
        except:
             return pd.DataFrame()

    def execute(self):
        # Recupera Sentiment GLOBALE (Macro)
        df_global_sent = self.get_sentiment_data("GLOBAL")
        
        # Baseline per scaling usando un ticker di punta (es. AAPL)
        df_sent_base = self.get_sentiment_data("AAPL")
        df_base = FeatureEngine.process_v7_features(self.get_market_data("AAPL"), self.macro, df_sent_base, df_global_sent)
        t_feat, m_feat, s_feat = FeatureEngine.extract_v7_features(df_base, macro_labels=MACRO_LABELS_ORDERED)
        
        scaler_t = StandardScaler().fit(t_feat)
        scaler_m = StandardScaler().fit(m_feat)
        scaler_s = StandardScaler().fit(s_feat)

        tf.keras.backend.clear_session()
        model = get_model("7.0", self.model_path, shape_t=(LOOKBACK_DAYS, t_feat.shape[1]), shape_m=(LOOKBACK_DAYS, m_feat.shape[1]), shape_s=(LOOKBACK_DAYS, s_feat.shape[1]))

        operazioni_fatte = []
        balance_str = ""

        for ticker in tqdm(TARGET_TICKERS_AZIONARIO, desc="🎯 Elaborazione V7 Azioni"):
            df_sent = self.get_sentiment_data(ticker)
            df_target = FeatureEngine.process_v7_features(self.get_market_data(ticker), self.macro, df_sent, df_global_sent)
            
            if len(df_target) < LOOKBACK_DAYS + 1: continue

            prezzo_oggi = df_target['prezzo'].iloc[-1]
            atr_oggi = max(df_target['ATRr_14'].iloc[-1], 1e-9)
            trend_rialzista = df_target['Dist_SMA200'].iloc[-1] > 0

            t_feat_raw, m_feat_raw, s_feat_raw = FeatureEngine.extract_v7_features(df_target, macro_labels=MACRO_LABELS_ORDERED)
            X_t = np.array([scaler_t.transform(t_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)
            X_m = np.array([scaler_m.transform(m_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)
            X_s = np.array([scaler_s.transform(s_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)

            # Predizione The Triple Brain
            delta_p = model.predict([X_t, X_m, X_s], verbose=0)[0][0] - 0.50
            state = self._load_state(ticker)

            if state['pos'] != 0:
                rend_pct = ((prezzo_oggi - state['entry']) / state['entry']) * (1 if state['pos'] == 1 else -1)
                if rend_pct <= -0.06:
                    state['pos'] = 0; state['invested_ratio'] = 0.0; state['qty'] = 0

            if state['pos'] == 0 and delta_p > 0.05 and trend_rialzista:
                ratio_aggiunto = min(1.0, TARGET_RISK / ((4.2 * atr_oggi) / prezzo_oggi)) * min(1.0, abs(delta_p) / 0.35)
                investimento_da_fare = CAPITALE_INIZIALE * ratio_aggiunto
                qty_ord = int(investimento_da_fare // prezzo_oggi)
                
                state['entry'] = prezzo_oggi; state['atr_entry'] = atr_oggi; state['invested_ratio'] = ratio_aggiunto
                state['pos'] = 1; state['qty'] = qty_ord; state['highest'] = prezzo_oggi; state['half_sold'] = False
                operazioni_fatte.append(f"🟢 <b>{ticker}</b>: BUY LONG SUGGERITO (${int(investimento_da_fare)})")

            self._save_state(ticker, state)

        trades_str = "\n".join(operazioni_fatte) if operazioni_fatte else ""
        self.set_report_data(balance_str=balance_str, trades_str=trades_str)

def run():
    StrategyV70().run()

if __name__ == "__main__":
    run()
