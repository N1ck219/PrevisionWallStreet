"""
strategy_v7_3.py — Strategia Intraday Split Brain V7.3.

Strategia di trading intraday in reale, derivata dalla V7.2, con:
    - Sizing dinamico dell'investimento ricalcolato sulla forza del segnale
    - SL/TP dinamici basati su ATR
    - Break-even status e Trailing Stop
    - Uscita parziale (50%) a TP e rilascio briglia (Trailing) per il resto
    - Gestione chiusura EOD
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from core.base_strategy import BaseStrategy
from core.config import (
    MODELS_DIR, DB_TRADES_V70, DB_MARKET, DB_MARKET_V70,
    TARGET_TICKERS_V70, MACRO_LABELS_ORDERED,
    ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
)
from core.data_manager import DataManager
from core.model_factory import get_model
from core.features import FeatureEngine

LOOKBACK = 60               # Finestra lookback (60 barre a 5-min = 5 ore)
HORIZON = 12                # Orizzonte previsione (12 barre a 5-min = 1 ora)
MAX_ALLOCATION = 0.20       # Max 20% per asset (v7.2 parameters)
SIGNAL_THRESHOLD = 0.06     # Soglia entrata
CLOSE_HOUR = 15             # Chiudi tutto dopo le 15:30 ET
CLOSE_MINUTE = 30

# Inizializzazione Alpaca
alpaca = None
if ALPACA_API_KEY_7 and ALPACA_SECRET_KEY_7:
    try:
        alpaca = TradingClient(ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7, paper=True)
    except Exception as e:
        print(f"⚠️ Errore connessione Alpaca V7.2: {e}")


class StrategyV73(BaseStrategy):
    bot_name = "V7.3 INTRADAY PRO (Dynamic Sizing)"
    db_trades_path = DB_TRADES_V70  # reuse same database file, using different tables
    db_market_path = DB_MARKET_V70
    model_path = os.path.join(MODELS_DIR, "intraday_brain_v7_0.h5")
    use_macro = True

    def setup_trades_db(self):
        super().setup_trades_db()
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS state_v73 (
            Ticker TEXT PRIMARY KEY,
            entry REAL,
            direction INTEGER,
            qty INTEGER,
            sl REAL,
            tp REAL,
            highest REAL,
            lowest REAL,
            status TEXT,
            stage INTEGER,
            entry_time TEXT,
            invested REAL
        )''')
        
        # Storico trade chiusi
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS trades_history_v73 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            direction INTEGER,
            entry_price REAL,
            exit_price REAL,
            qty INTEGER,
            pnl REAL,
            entry_time TEXT,
            exit_time TEXT,
            exit_reason TEXT
        )''')
        
        # Inizializza stato per ogni ticker
        for t in TARGET_TICKERS_V70:
            if not self.conn_trades.execute(
                "SELECT count(*) FROM state_v73 WHERE Ticker=?", (t,)
            ).fetchone()[0]:
                self.conn_trades.execute(
                    "INSERT INTO state_v73 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (t, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 'normal', 1, '', 0.0)
                )
        self.conn_trades.commit()

    def _load_state(self, ticker):
        row = self.conn_trades.execute(
            "SELECT entry, direction, qty, sl, tp, highest, lowest, status, stage, entry_time, invested FROM state_v73 WHERE Ticker=?",
            (ticker,)
        ).fetchone()
        return {
            'entry': row[0], 'direction': row[1], 'qty': row[2],
            'sl': row[3], 'tp': row[4], 'highest': row[5], 'lowest': row[6],
            'status': row[7], 'stage': row[8], 'entry_time': row[9], 'invested': row[10]
        }

    def _save_state(self, ticker, s):
        self.conn_trades.execute(
            "UPDATE state_v73 SET entry=?, direction=?, qty=?, sl=?, tp=?, highest=?, lowest=?, status=?, stage=?, entry_time=?, invested=? WHERE Ticker=?",
            (s['entry'], s['direction'], s['qty'], s['sl'], s['tp'], s['highest'], s['lowest'], s['status'], s['stage'], s['entry_time'], s['invested'], ticker)
        )
        self.conn_trades.commit()

    def _record_trade(self, ticker, direction, entry_price, exit_price, qty, pnl, entry_time, exit_time, reason):
        """Registra un trade chiuso nello storico."""
        self.conn_trades.execute(
            "INSERT INTO trades_history_v73 (ticker, direction, entry_price, exit_price, qty, pnl, entry_time, exit_time, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, direction, entry_price, exit_price, qty, pnl, entry_time, exit_time, reason)
        )
        self.conn_trades.commit()

    def _close_position(self, ticker, state, prezzo_uscita, reason):
        """Chiude una posizione e registra il trade."""
        pnl = (prezzo_uscita - state['entry']) * state['qty'] * state['direction']
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Registra nello storico
        self._record_trade(
            ticker, state['direction'], state['entry'], prezzo_uscita,
            state['qty'], pnl, state['entry_time'], now_str, reason
        )
        
        # Chiudi su Alpaca
        if alpaca:
            try:
                # Use standard alpaca call closely mapping close_position semantics
                alpaca.close_position(ticker)
            except Exception as e:
                print(f"⚠️ Errore chiusura Alpaca {ticker}: {e}")
        
        # Reset stato
        state['entry'] = 0.0
        state['direction'] = 0
        state['qty'] = 0
        state['sl'] = 0.0
        state['tp'] = 0.0
        state['highest'] = 0.0
        state['lowest'] = 0.0
        state['status'] = 'normal'
        state['stage'] = 1
        state['entry_time'] = ''
        state['invested'] = 0.0
        self._save_state(ticker, state)
        
        emoji = "💚" if pnl > 0 else "💔"
        return f"{emoji} <b>{ticker}</b>: CHIUSO ({reason}) PnL: ${pnl:+,.2f}"

    def _is_market_open(self):
        """Verifica se il mercato è aperto (orario ET)."""
        now_et = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))
        weekday = now_et.weekday()
        if weekday >= 5:  # Weekend
            return False, now_et
        
        hour, minute = now_et.hour, now_et.minute
        total_min = hour * 60 + minute
        
        # Mercato aperto: 9:30 - 16:00 ET
        if 9 * 60 + 30 <= total_min < 16 * 60:
            return True, now_et
        return False, now_et

    def _should_close_only(self, now_et):
        """Verifica se siamo nell'ultima mezz'ora (solo chiusure, no nuove aperture)."""
        total_min = now_et.hour * 60 + now_et.minute
        return total_min >= CLOSE_HOUR * 60 + CLOSE_MINUTE

    def _is_5min_boundary(self, now_et):
        """Verifica se siamo su un confine a 5 minuti (per inferenza modello)."""
        return now_et.minute % 5 == 0

    def _log_predictions(self, results):
        """Salva le previsioni in un file CSV dedicato per il monitoraggio."""
        csv_path = os.path.join(os.path.dirname(self.db_market_path), "..", "reports", "v7_3_predictions_log.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        file_exists = os.path.isfile(csv_path)
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(csv_path, 'a', encoding='utf-8') as f:
            if not file_exists:
                f.write("Datetime,Ticker,Price,Prediction,Delta,Threshold_Hit\n")
            
            for r in results:
                hit = "YES" if abs(r['delta']) > SIGNAL_THRESHOLD else "NO"
                f.write(f"{now_str},{r['ticker']},{r['price']:.2f},{r['pred']:.4f},{r['delta']:.4f},{hit}\n")

    def execute(self):
        """Logica principale della strategia intraday V7.2."""
        market_open, now_et = self._is_market_open()
        
        if not market_open:
            print(f"⏸️ Mercato chiuso (ET: {now_et.strftime('%H:%M')}). Nessuna azione.")
            self.set_report_data(
                extra_str=f"Mercato chiuso (ET: {now_et.strftime('%H:%M %Z')})"
            )
            return
        
        close_only = self._should_close_only(now_et)
        is_5min = self._is_5min_boundary(now_et)
        
        if close_only:
            print(f"⏰ Ultima mezz'ora — solo chiusure posizioni aperte")
        elif not is_5min:
            print(f"⏳ Non siamo su un confine a 5 min — solo monitoraggio SL/TP")
        
        # Prepara il modello (solo se serve per segnali)
        model = None
        n_tech = 11  # fallback
        n_macro = 7  # fallback
        
        if is_5min and not close_only:
            if not hasattr(self, '_model') or self._model is None:
                for ticker in TARGET_TICKERS_V70:
                    df_raw = DataManager.get_cached_intraday_data(
                        ticker, self.conn_market, ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
                    )
                    if len(df_raw) > LOOKBACK * 5 + 500:
                        df_5min = DataManager.resample_to_5min(df_raw)
                        df_feat = FeatureEngine.process_intraday_features(df_5min, self.macro)
                        t_sample, m_sample = FeatureEngine.extract_intraday_features(df_feat, macro_labels=MACRO_LABELS_ORDERED)
                        n_tech = t_sample.shape[1]
                        n_macro = m_sample.shape[1]
                        break
                
                tf.keras.backend.clear_session()
                self._model = get_model("7.0", self.model_path, shape_t=(LOOKBACK, n_tech), shape_m=(LOOKBACK, n_macro))
        
        # Portafoglio Alpaca
        try:
            acc = alpaca.get_account()
            portfolio_value = float(acc.equity)
            cash_disponibile = float(acc.buying_power)
        except:
            portfolio_value = 100_000.0
            cash_disponibile = 100_000.0
        
        operazioni_fatte = []
        logs_chiusure = []
        
        batch_tickers = []
        batch_X_t = []
        batch_X_m = []
        batch_meta = {}
        
        for ticker in tqdm(TARGET_TICKERS_V70, desc="🎯 Analisi Intraday V7.3"):
            state = self._load_state(ticker)
            
            # Sincronizza i dati da Alpaca (scarica solo i mancanti, molto veloce)
            DataManager.get_cached_intraday_data(
                ticker, self.conn_market, ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
            )
            
            # QUERY TURBO: Legge solo le ultime 1000 candele (circa 16 ore di trading)
            # Invece di caricare mesi di storico, carichiamo solo lo stretto necessario per le feature e SL/TP.
            df_raw = pd.read_sql_query(
                "SELECT * FROM (SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker=? ORDER BY Datetime DESC LIMIT 1000) ORDER BY Datetime ASC",
                self.conn_market, params=(ticker,)
            )

            if len(df_raw) < LOOKBACK * 5:
                continue
            
            prezzo_corrente = df_raw['Close'].iloc[-1]
            
            # ── Gestione posizioni aperte (monitoraggio iterativo per max precisione SL/TP) ──
            if state['direction'] != 0:
                rend_pct = ((prezzo_corrente - state['entry']) / state['entry']) * state['direction']
                
                # Check EOD
                if close_only:
                    msg = self._close_position(ticker, state, prezzo_corrente, 'EOD')
                    logs_chiusure.append(msg)
                    continue

                # Aggiorna highest/lowest
                if state['direction'] == 1:
                    if prezzo_corrente > state['highest']: state['highest'] = prezzo_corrente
                    max_rend = (state['highest'] - state['entry']) / state['entry']
                else:
                    if prezzo_corrente < state['lowest']: state['lowest'] = prezzo_corrente
                    max_rend = (state['entry'] - state['lowest']) / state['entry']

                # Avanzamento status (Dynamic logic V7.2)
                if state['status'] == 'normal' and max_rend >= 0.01:
                    state['status'] = 'break_even'
                if state['status'] in ['normal', 'break_even'] and max_rend >= 0.015:
                    state['status'] = 'trailing'

                # Calcolo Stop Loss dinamico
                if state['status'] == 'normal':
                    sl_price = state['entry'] * (1 - state['sl'] * state['direction'])
                elif state['status'] == 'break_even':
                    sl_price = state['entry'] * (1 + 0.001 * state['direction'])
                else: # trailing
                    sl_price = state['highest'] * (1 - 0.008) if state['direction'] == 1 else state['lowest'] * (1 + 0.008)

                is_sl = (state['direction'] == 1 and prezzo_corrente <= sl_price) or \
                        (state['direction'] == -1 and prezzo_corrente >= sl_price)

                if is_sl:
                    reason = 'TRAIL' if state['status'] == 'trailing' else ('BE' if state['status'] == 'break_even' else 'SL')
                    msg = self._close_position(ticker, state, prezzo_corrente, reason)
                    logs_chiusure.append(msg)
                    continue

                # Take Profit Parziale (Stage 1 -> 2)
                if state['stage'] == 1 and rend_pct >= state['tp']:
                    sell_qty = state['qty'] // 2
                    if sell_qty > 0:
                        if alpaca:
                            try:
                                close_side = OrderSide.SELL if state['direction'] == 1 else OrderSide.BUY
                                alpaca.submit_order(MarketOrderRequest(
                                    symbol=ticker, qty=sell_qty,
                                    side=close_side, time_in_force=TimeInForce.DAY
                                ))
                            except Exception as e:
                                print(f"⚠️ Errore parziale Alpaca {ticker}: {e}")

                        # Aggiorniamo info PnL parziale
                        pnl = (prezzo_corrente - state['entry']) * sell_qty * state['direction']
                        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self._record_trade(
                            ticker, state['direction'], state['entry'], prezzo_corrente,
                            sell_qty, pnl, state['entry_time'], now_str, 'TP_50%'
                        )

                        state['qty'] -= sell_qty
                        state['stage'] = 2
                        state['status'] = 'trailing'
                        
                        emoji = "🟢" if pnl > 0 else "🔴"
                        logs_chiusure.append(f"{emoji} <b>{ticker}</b>: TP_50% PnL: ${pnl:+,.2f}")
                        
                self._save_state(ticker, state)
            
            # ── Generazione segnali (riempimento batch IA) ──
            if close_only or state['direction'] != 0 or not is_5min or not hasattr(self, '_model'):
                continue
            
            df_5min = DataManager.resample_to_5min(df_raw)
            df_feat = FeatureEngine.process_intraday_features(df_5min, self.macro)
            
            if len(df_feat) < LOOKBACK + 1:
                continue
            
            t_feat, m_feat = FeatureEngine.extract_intraday_features(df_feat, macro_labels=MACRO_LABELS_ORDERED)
            
            scaler_t = StandardScaler().fit(t_feat)
            scaler_m = StandardScaler().fit(m_feat)
            
            
            X_t = np.array(scaler_t.transform(t_feat)[-LOOKBACK:], dtype=np.float32)
            X_m = np.array(scaler_m.transform(m_feat)[-LOOKBACK:], dtype=np.float32)
            
            if len(X_t) < LOOKBACK:
                continue
            
            batch_tickers.append(ticker)
            batch_X_t.append(X_t)
            batch_X_m.append(X_m)
            batch_meta[ticker] = {
                'prezzo_corrente': prezzo_corrente,
                'state': state,
                'atr_val': df_feat['ATRr_14'].iloc[-1] if 'ATRr_14' in df_feat.columns else 0.0
            }
            
        # ── Inferenza IA di Gruppo (Batch Prediction) ──
        if batch_tickers and hasattr(self, '_model'):
            X_t_arr = np.array(batch_X_t)
            X_m_arr = np.array(batch_X_m)
            preds = self._model.predict([X_t_arr, X_m_arr], verbose=0).flatten()
            
            prediction_results = []
            print(f"\n🧠 [PREVISIONI IA - {datetime.datetime.now().strftime('%H:%M:%S')}]")
            print(f"{'Ticker':<8} | {'Prezzo':<10} | {'Predizione':<10} | {'Delta':<8} | {'Segnale'}")
            print("-" * 60)

            for ticker, pred in zip(batch_tickers, preds):
                delta = pred - 0.50
                meta = batch_meta[ticker]
                prezzo_corrente = meta['prezzo_corrente']
                state = meta['state']
                atr_val = meta['atr_val']
                
                hit = "🚀" if delta > SIGNAL_THRESHOLD else ("📉" if delta < -SIGNAL_THRESHOLD else "➖")
                print(f"{ticker:<8} | {prezzo_corrente:<10.2f} | {pred:<10.4f} | {delta:<+8.4f} | {hit}")
                
                prediction_results.append({
                    'ticker': ticker, 'price': prezzo_corrente, 'pred': pred, 'delta': delta
                })

                if abs(delta) > SIGNAL_THRESHOLD:
                    direction = 1 if delta > 0 else -1
                    side = OrderSide.BUY if direction == 1 else OrderSide.SELL
                    side_label = "BUY LONG" if direction == 1 else "SELL SHORT"
                    
                    # Sizing Dinamico in base alla forza del segnale
                    abs_delta = abs(delta)
                    if abs_delta >= 0.12:
                        alloc = 0.20
                    elif abs_delta >= 0.08:
                        alloc = 0.15
                    else:
                        alloc = 0.10
                        
                    budget = min(portfolio_value * alloc, cash_disponibile * 0.95)
                    qty = int(budget // prezzo_corrente)
                    
                    if qty > 0 and budget > 100:
                        baseline_sl = (2.0 * atr_val / prezzo_corrente) if atr_val > 0 else 0.02
                        dynamic_sl = max(0.005, min(baseline_sl, 0.035))
                        dynamic_tp = dynamic_sl * 1.5

                        try:
                            if alpaca:
                                alpaca.submit_order(MarketOrderRequest(
                                    symbol=ticker, qty=qty,
                                    side=side, time_in_force=TimeInForce.DAY
                                ))
                            
                            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            investimento = qty * prezzo_corrente
                            
                            state['entry'] = prezzo_corrente
                            state['direction'] = direction
                            state['qty'] = qty
                            state['sl'] = dynamic_sl
                            state['tp'] = dynamic_tp
                            state['highest'] = prezzo_corrente
                            state['lowest'] = prezzo_corrente
                            state['status'] = 'normal'
                            state['stage'] = 1
                            state['entry_time'] = now_str
                            state['invested'] = investimento
                            self._save_state(ticker, state)
                            
                            cash_disponibile -= investimento
                            
                            emoji = "🟢" if direction == 1 else "🔴"
                            conf_pct = abs(delta) * 200
                            operazioni_fatte.append(
                                f"{emoji} <b>{ticker}</b>: {side_label} ${int(investimento)} (SL {dynamic_sl*100:.1f}%) "
                                f"(conf: {conf_pct:.0f}%)"
                            )
                        except Exception as e:
                            print(f"❌ Errore ordine {ticker}: {e}")
            
            # Salva lo storico delle previsioni
            self._log_predictions(prediction_results)
            print("-" * 60)
        
        # ── Report ──
        win_rate_str = ""
        try:
            total = self.conn_trades.execute("SELECT count(*) FROM trades_history_v73").fetchone()[0]
            wins = self.conn_trades.execute("SELECT count(*) FROM trades_history_v73 WHERE pnl > 0").fetchone()[0]
            if total > 0:
                win_rate_str = f"{wins}/{total} ({wins/total*100:.1f}%)"
        except:
            pass
        
        trades_str = "\n".join(operazioni_fatte) if operazioni_fatte else ""
        logs_str = "\n".join(logs_chiusure) if logs_chiusure else ""
        
        pos_aperte = sum(1 for t in TARGET_TICKERS_V70 if self._load_state(t)['direction'] != 0)
        
        mode = 'SOLO CHIUSURE' if close_only else ('SEGNALI 5min' if is_5min else 'MONITORAGGIO SL/TP (V7.3)')
        
        self.set_report_data(
            balance_str=f"${portfolio_value:,.2f}",
            win_rate_str=win_rate_str,
            trades_str=trades_str,
            logs_str=logs_str,
            extra_str=f"Posizioni aperte: {pos_aperte}/{len(TARGET_TICKERS_V70)} | "
                      f"Modalità: {mode}"
        )

    def send_report(self):
        """Invia il report riassuntivo solo a fine giornata (alle 16:00 ET)."""
        now_et = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))
        total_min = now_et.hour * 60 + now_et.minute
        
        # Invia solo alle 16:00 ET (fine giornata EOD) o oltre
        if total_min == 960:
            today_str = datetime.datetime.now().strftime('%Y-%m-%d')
            # Raccogli tutti i trade di oggi
            import sqlite3
            try:
                # Creiamo una connessione temporanea perché _cleanup() l'ha già chiusa
                conn_tmp = sqlite3.connect(self.db_trades_path)
                cursor = conn_tmp.execute(
                    "SELECT ticker, direction, pnl, exit_reason FROM trades_history_v73 WHERE exit_time LIKE ?",
                    (f"{today_str}%",)
                )
                trades_oggi = cursor.fetchall()
                conn_tmp.close()
                
                if trades_oggi:
                    riassunto = ["<b>📊 RIASSUNTO OPERAZIONI ODIERNE:</b>"]
                    tot_pnl = 0
                    for t, d, pnl, reason in trades_oggi:
                        emoji = "💚" if pnl > 0 else "💔"
                        dir_str = "LONG" if d == 1 else "SHORT"
                        riassunto.append(f"{emoji} <b>{t}</b> {dir_str} ({reason}) PnL: ${pnl:+,.2f}")
                        tot_pnl += pnl
                    
                    riassunto.append(f"\n<b>💰 PnL Totale Giornata: ${tot_pnl:+,.2f}</b>")
                    self._report_data['trades_str'] = "\n".join(riassunto)
                    self._report_data['logs_str'] = "" # puliamo i log intermedi
                else:
                    self._report_data['trades_str'] = "📭 Nessuna operazione chiusa oggi."
                    self._report_data['logs_str'] = ""
            except Exception as e:
                print(f"Errore generazione recap: {e}")
                
            super().send_report()


def run():
    StrategyV73().run()


if __name__ == "__main__":
    run()
