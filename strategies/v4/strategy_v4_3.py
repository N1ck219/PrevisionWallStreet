import os
import time
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import TakeProfitRequest, StopLossRequest

from strategies.base_strategy import BaseStrategy
from core.config import MODELS_DIR, DB_TRADES_V43, TARGET_TICKERS_V43, BASE_TICKERS_V43, ALPACA_API_KEY, ALPACA_SECRET_KEY, MACRO_MAP
from core.models.model_factory import get_model, build_v4_3_model
from core.data.data_manager import DataManager
from core.data.features import FeatureEngine

LOOKBACK_DAYS = 60
CAPITALE_INIZIALE = 10000.0
FEE_TRANSAZIONE = 0.0015
TARGET_RISK_V43 = 0.20

alpaca = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    try: alpaca = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    except: pass


class StrategyV43(BaseStrategy):
    bot_name = "V4.3 HEDGE FUND LIVE SCANNER"
    db_trades_path = DB_TRADES_V43
    model_path = os.path.join(MODELS_DIR, "base_brain_v4.h5")

    FEATURE_COLS = ['ret', 'vol_ret', 'nasdaq_ret', 'vix_ret', 'tnx_ret', 'soxx_ret', 'gld_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA200', 'OBV_ret']

    def setup_trades_db(self):
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS portfolio_v42 (Ticker TEXT PRIMARY KEY, Capital REAL, Last_Position REAL, Last_Size REAL, Last_Price REAL, Last_Date TEXT, High_Mem REAL, Low_Mem REAL)''')
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS history_v42 (Date TEXT, Ticker TEXT, Bot_Return REAL, Win_Loss TEXT)''')
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS portfolio_v43 (Ticker TEXT PRIMARY KEY, Capital REAL, Last_Position REAL, Last_Size REAL, Last_Price REAL, Last_Date TEXT, High_Mem REAL, Low_Mem REAL)''')
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS history_v43 (Date TEXT, Ticker TEXT, Bot_Return REAL, Win_Loss TEXT)''')
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS benchmark_bh (Ticker TEXT PRIMARY KEY, Initial_Price REAL, Start_Date TEXT)''')
        for t in TARGET_TICKERS_V43:
            if not self.conn_trades.execute("SELECT count(*) FROM portfolio_v42 WHERE Ticker=?", (t,)).fetchone()[0]:
                self.conn_trades.execute("INSERT INTO portfolio_v42 VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (t, CAPITALE_INIZIALE, 0.0, 0.0, 0.0, '2000-01-01', 0.0, 999999.0))
            if not self.conn_trades.execute("SELECT count(*) FROM portfolio_v43 WHERE Ticker=?", (t,)).fetchone()[0]:
                self.conn_trades.execute("INSERT INTO portfolio_v43 VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (t, CAPITALE_INIZIALE, 0.0, 0.0, 0.0, '2000-01-01', 0.0, 999999.0))
        self.conn_trades.commit()

    def _process_logic(self, ticker, table_port, table_hist, pos, size, cap, last_price, last_date, high_m, low_m, prezzo_oggi, massimo_oggi, minimo_oggi, atr_oggi, data_oggi_db):
        chiuso = False; rendimento = 0; motivo = ""
        if last_date != '2000-01-01' and last_date != data_oggi_db and pos != 0:
            if pos == 1 and massimo_oggi > high_m: high_m = massimo_oggi
            elif pos == -1 and minimo_oggi < low_m: low_m = minimo_oggi
            if pos == 1:
                sl_price = high_m - (4.2 * atr_oggi)
                if minimo_oggi <= sl_price: rendimento = ((sl_price - last_price) / last_price) * size; chiuso = True; motivo = "Trailing Stop LONG"
            elif pos == -1:
                sl_price = low_m + (3.0 * atr_oggi)
                if massimo_oggi >= sl_price: rendimento = ((last_price - sl_price) / last_price) * size; chiuso = True; motivo = "Trailing Stop SHORT"
            if not chiuso:
                rend_mkt = (prezzo_oggi - last_price) / last_price
                if pos == 1: rendimento = rend_mkt * size
                elif pos == -1: rendimento = -rend_mkt * size
                motivo = "Chiusura Giornaliera"
            nuovo_cap = cap * (1 + rendimento)
            esito = "✅" if rendimento > 0 else ("❌" if rendimento < 0 else "➖")
            self.conn_trades.execute(f"INSERT INTO {table_hist} VALUES (?, ?, ?, ?)", (data_oggi_db, ticker, rendimento, esito))
            self.conn_trades.execute(f"UPDATE {table_port} SET Capital=? WHERE Ticker=?", (nuovo_cap, ticker))
            self.conn_trades.commit()
            pos_str = "LONG" if pos == 1 else "SHORT"
            report = f"{ticker}: {rendimento*100:+.2f}% ({pos_str}) {esito} <i>[{motivo}]</i>\n"
            if chiuso: pos = 0; size = 0.0
            return nuovo_cap, pos, size, high_m, low_m, report
        elif pos == 0 and last_date != data_oggi_db and last_date != '2000-01-01':
            return cap, pos, size, high_m, low_m, f"{ticker}: Cash (Protetti) 🛡️\n"
        return cap, pos, size, high_m, low_m, ""

    def _calc_win_rate(self, table, ticker):
        storico = self.conn_trades.execute(f"SELECT Win_Loss FROM {table} WHERE Ticker=?", (ticker,)).fetchall()
        tot = sum(1 for r in storico if r[0] in ["✅", "❌"])
        vinte = sum(1 for r in storico if r[0] == "✅")
        return f"{int((vinte/tot)*100)}% ({vinte}/{tot})" if tot > 0 else "N/A"

    def _train_base_model(self):
        """Addestra il modello base su tutti i BASE_TICKERS_V43."""
        print("🧠 Modello Base non trovato. Generazione tramite Factory...")
        all_X, all_y = [], []
        for bt in tqdm(BASE_TICKERS_V43):
            df_b = FeatureEngine.process_stock_features(self.get_market_data(bt), self.macro)
            df_b['Target'] = (df_b['prezzo'].shift(-5) > df_b['prezzo']).astype(int)
            df_b.dropna(inplace=True)
            if len(df_b) < 100: continue
            feat = StandardScaler().fit_transform(df_b[self.FEATURE_COLS].values)
            for i in range(LOOKBACK_DAYS, len(feat)):
                all_X.append(feat[i-LOOKBACK_DAYS:i])
                all_y.append(df_b['Target'].iloc[i])
        base_model = build_v4_3_model((LOOKBACK_DAYS, len(self.FEATURE_COLS)))
        base_model.fit(np.array(all_X), np.array(all_y), epochs=15, batch_size=256, verbose=0, callbacks=[TqdmCallback(verbose=0)])
        base_model.save_weights(self.model_path)

    def execute(self):
        # Sync dati
        macro_tickers = ['QQQ', '^VIX', '^TNX', 'SOXX', 'GLD']
        for t in tqdm(TARGET_TICKERS_V43 + macro_tickers, desc="📥 Sync DB"):
            self.get_market_data(t)

        # Addestra modello base se mancante
        if not os.path.exists(self.model_path):
            self._train_base_model()

        report_ieri_43 = ""; report_domani = ""; stats_azione = ""
        posizioni_alpaca = {p.symbol: p for p in alpaca.get_all_positions()} if alpaca else {}

        for ticker in TARGET_TICKERS_V43:
            df_target = FeatureEngine.process_stock_features(self.get_market_data(ticker), self.macro)
            df_target['Target'] = (df_target['prezzo'].shift(-5) > df_target['prezzo']).astype(int)
            df_target.dropna(subset=['prezzo', 'Target'] + self.FEATURE_COLS, inplace=True)
            if len(df_target) < LOOKBACK_DAYS + 1: continue

            c_42, p_42, s_42, lp_42, ld_42, h_42, l_42 = self.conn_trades.execute("SELECT Capital, Last_Position, Last_Size, Last_Price, Last_Date, High_Mem, Low_Mem FROM portfolio_v42 WHERE Ticker=?", (ticker,)).fetchone()
            c_43, p_43, s_43, lp_43, ld_43, h_43, l_43 = self.conn_trades.execute("SELECT Capital, Last_Position, Last_Size, Last_Price, Last_Date, High_Mem, Low_Mem FROM portfolio_v43 WHERE Ticker=?", (ticker,)).fetchone()

            prezzo_oggi, massimo_oggi, minimo_oggi = df_target['prezzo'].iloc[-1], df_target['High'].iloc[-1], df_target['Low'].iloc[-1]
            atr_oggi = max(df_target['ATRr_14'].iloc[-1], 1e-9)
            data_oggi_db = str(df_target['data'].iloc[-1])
            trend_rialzista = df_target['Dist_SMA200'].iloc[-1] > 0

            bh_record = self.conn_trades.execute("SELECT Initial_Price FROM benchmark_bh WHERE Ticker=?", (ticker,)).fetchone()
            if not bh_record:
                self.conn_trades.execute("INSERT INTO benchmark_bh VALUES (?, ?, ?)", (ticker, prezzo_oggi, data_oggi_db))
                self.conn_trades.commit(); bh_initial = prezzo_oggi
            else: bh_initial = bh_record[0]

            c_42, p_42, s_42, h_42, l_42, rep_42 = self._process_logic(ticker, "portfolio_v42", "history_v42", p_42, s_42, c_42, lp_42, ld_42, h_42, l_42, prezzo_oggi, massimo_oggi, minimo_oggi, atr_oggi, data_oggi_db)
            c_43, p_43, s_43, h_43, l_43, rep_43 = self._process_logic(ticker, "portfolio_v43", "history_v43", p_43, s_43, c_43, lp_43, ld_43, h_43, l_43, prezzo_oggi, massimo_oggi, minimo_oggi, atr_oggi, data_oggi_db)
            report_ieri_43 += rep_43

            feat_scaled = StandardScaler().fit_transform(df_target[self.FEATURE_COLS].values)
            X_all = np.array([feat_scaled[i-LOOKBACK_DAYS:i] for i in range(LOOKBACK_DAYS, len(feat_scaled) + 1)])
            X_predict = X_all[-1:]

            tf.keras.backend.clear_session()
            live_model = get_model("4.3", self.model_path, input_shape=(LOOKBACK_DAYS, len(self.FEATURE_COLS)))
            live_model.get_layer("lstm_base").trainable = False

            y_train = df_target['Target'].values[LOOKBACK_DAYS:-1]
            X_train = X_all[:-1]
            if len(X_train) > 100: live_model.fit(X_train[-200:], np.array(y_train[-200:]), batch_size=32, epochs=5, verbose=0)

            delta_p = live_model.predict(X_predict, verbose=0)[0][0] - 0.50
            MARGINE = 0.05

            if delta_p > MARGINE: np_42 = 1; ns_42 = 1.0 if trend_rialzista else 0.4
            elif delta_p < -MARGINE: np_42 = -1 if not trend_rialzista else 0; ns_42 = 1.0 if not trend_rialzista else 0.0
            else: np_42 = 0; ns_42 = 0.0

            if abs((np_42 * ns_42) - (p_42 * s_42)) > 0:
                c_42 = c_42 * (1 - FEE_TRANSAZIONE)
                if np_42 == 1: h_42 = prezzo_oggi; l_42 = 999999.0
                elif np_42 == -1: l_42 = prezzo_oggi; h_42 = 0.0

            if delta_p > MARGINE: np_43, ns_43, txt_43 = 1, min(1.0, TARGET_RISK_V43 / ((4.2 * atr_oggi) / prezzo_oggi)), "🟢 LONG"
            elif delta_p < -MARGINE: np_43, ns_43, txt_43 = -1, min(1.0, TARGET_RISK_V43 / ((3.0 * atr_oggi) / prezzo_oggi)), "🔴 SHORT"
            else: np_43, ns_43, txt_43 = 0, 0.0, "⚪ CASH"

            cambio_stato_alpaca = abs((np_43 * ns_43) - (p_43 * s_43)) > 0
            if cambio_stato_alpaca:
                c_43 = c_43 * (1 - FEE_TRANSAZIONE)
                if np_43 == 1: h_43 = prezzo_oggi; l_43 = 999999.0
                elif np_43 == -1: l_43 = prezzo_oggi; h_43 = 0.0

            self.conn_trades.execute("UPDATE portfolio_v42 SET Last_Position=?, Last_Size=?, Last_Price=?, Last_Date=?, Capital=?, High_Mem=?, Low_Mem=? WHERE Ticker=?", (np_42, ns_42, prezzo_oggi, data_oggi_db, c_42, h_42, l_42, ticker))
            self.conn_trades.execute("UPDATE portfolio_v43 SET Last_Position=?, Last_Size=?, Last_Price=?, Last_Date=?, Capital=?, High_Mem=?, Low_Mem=? WHERE Ticker=?", (np_43, ns_43, prezzo_oggi, data_oggi_db, c_43, h_43, l_43, ticker))
            self.conn_trades.commit()

            stats_azione += f"<b>{ticker}:</b> V4.3 [{self._calc_win_rate('history_v43', ticker)}]\\n"
            report_domani += f"<b>{ticker}</b> <i>(B&H: {((prezzo_oggi - bh_initial) / bh_initial) * 100:+.1f}%)</i>\n└ 🎯 V4.3 (AI): {txt_43} ({int(ns_43*100)}%)\n"

            if alpaca:
                try:
                    if np_43 == 0 or (np_43 != p_43 and p_43 != 0):
                        alpaca.close_position(ticker); alpaca.cancel_orders(symbol=ticker); report_domani += f"     🧹 <i>Alpaca Chiuso.</i>\n"; time.sleep(1)
                    if cambio_stato_alpaca and np_43 != 0:
                        quantita_azioni = int((c_43 * ns_43) // prezzo_oggi)
                        if quantita_azioni > 0:
                            lato = OrderSide.BUY if np_43 == 1 else OrderSide.SELL
                            stop_price = round(prezzo_oggi - (4.2 * atr_oggi), 2) if np_43 == 1 else round(prezzo_oggi + (3.0 * atr_oggi), 2)
                            take_profit = round(prezzo_oggi + (8.0 * atr_oggi), 2) if np_43 == 1 else round(prezzo_oggi - (6.0 * atr_oggi), 2)
                            alpaca.submit_order(MarketOrderRequest(symbol=ticker, qty=quantita_azioni, side=lato, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, stop_loss=StopLossRequest(stop_price=stop_price), take_profit=TakeProfitRequest(limit_price=take_profit)))
                            report_domani += f"     ⚡ <b>Azione Alpaca Oggi:</b> {lato.name} {quantita_azioni} az. | 🛡️ SL @ ${stop_price}\n"
                except Exception as e: report_domani += f"     ❌ <i>Alpaca Errore: {e}</i>\n"

        # Report
        balance_str = f"${float(alpaca.get_account().equity):,.2f}" if alpaca else ""
        self.set_report_data(
            balance_str=balance_str,
            win_rate_str=stats_azione,
            trades_str=report_domani,
            logs_str=report_ieri_43 or "Nessuna."
        )


# ── Funzione compatibile con run_stock.py ──
def run():
    StrategyV43().run()

if __name__ == "__main__":
    run()