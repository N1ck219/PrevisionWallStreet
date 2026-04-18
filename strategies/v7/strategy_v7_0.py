"""
strategy_v7_0.py — Strategia Intraday Split Brain V7.0.

Strategia di trading intraday che:
    - Scarica dati a 1 minuto da Alpaca
    - Li resampla a 5 minuti per l'input del modello
    - Produce previsioni sull'orizzonte di 1 ora (12 barre a 5 min)
    - Monitora SL/TP sulla risoluzione a 1 minuto per massima reattività

Logica:
    - Opera solo durante le ore di mercato (9:30 - 15:30 ET)
    - Stop loss 2%, Take profit 1.5%
    - Chiusura forzata di tutte le posizioni a fine sessione
    - Max 1 posizione per ticker simultaneamente
    - Il modello viene interrogato solo alla chiusura di ogni candela a 5 min
"""

import os
import time
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from strategies.base_strategy import BaseStrategy
from core.config import (
    MODELS_DIR, DB_TRADES_V70, DB_MARKET, DB_MARKET_V70,
    TARGET_TICKERS_V70, MACRO_LABELS_ORDERED,
    ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
)
from core.data.data_manager import DataManager
from core.models.model_factory import get_model
from core.data.features import FeatureEngine

LOOKBACK = 60               # Finestra lookback (60 barre a 5-min = 5 ore)
HORIZON = 12                # Orizzonte previsione (12 barre a 5-min = 1 ora)
STOP_LOSS_PCT = 0.02        # Stop loss 2%
TAKE_PROFIT_PCT = 0.015     # Take profit 1.5%
MAX_ALLOCATION = 0.15       # Max 15% per asset
SIGNAL_THRESHOLD = 0.06     # Soglia entrata
CLOSE_HOUR = 15             # Chiudi tutto dopo le 15:30 ET
CLOSE_MINUTE = 30

# Inizializzazione Alpaca
alpaca = None
if ALPACA_API_KEY_7 and ALPACA_SECRET_KEY_7:
    try:
        alpaca = TradingClient(ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7, paper=True)
    except Exception as e:
        print(f"⚠️ Errore connessione Alpaca V7.0: {e}")


class StrategyV70(BaseStrategy):
    bot_name = "V7.0 INTRADAY SNIPER (Split Brain 5min)"
    db_trades_path = DB_TRADES_V70
    db_market_path = DB_MARKET_V70
    model_path = os.path.join(MODELS_DIR, "intraday_brain_v7_0.h5")
    use_macro = True

    def setup_trades_db(self):
        # ... (unchanged)
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS state_v70 (
            Ticker TEXT PRIMARY KEY,
            entry REAL,
            direction INTEGER,
            qty INTEGER,
            stop_loss REAL,
            take_profit REAL,
            entry_time TEXT,
            invested REAL
        )''')
        
        # Storico trade chiusi
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS trades_history_v70 (
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
                "SELECT count(*) FROM state_v70 WHERE Ticker=?", (t,)
            ).fetchone()[0]:
                self.conn_trades.execute(
                    "INSERT INTO state_v70 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (t, 0.0, 0, 0, 0.0, 0.0, '', 0.0)
                )
        self.conn_trades.commit()

    def _load_state(self, ticker):
        row = self.conn_trades.execute(
            "SELECT entry, direction, qty, stop_loss, take_profit, entry_time, invested FROM state_v70 WHERE Ticker=?",
            (ticker,)
        ).fetchone()
        return {
            'entry': row[0], 'direction': row[1], 'qty': row[2],
            'stop_loss': row[3], 'take_profit': row[4],
            'entry_time': row[5], 'invested': row[6]
        }

    def _save_state(self, ticker, s):
        self.conn_trades.execute(
            "UPDATE state_v70 SET entry=?, direction=?, qty=?, stop_loss=?, take_profit=?, entry_time=?, invested=? WHERE Ticker=?",
            (s['entry'], s['direction'], s['qty'], s['stop_loss'], s['take_profit'], s['entry_time'], s['invested'], ticker)
        )
        self.conn_trades.commit()

    def _record_trade(self, ticker, direction, entry_price, exit_price, qty, pnl, entry_time, exit_time, reason):
        """Registra un trade chiuso nello storico."""
        self.conn_trades.execute(
            "INSERT INTO trades_history_v70 (ticker, direction, entry_price, exit_price, qty, pnl, entry_time, exit_time, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                alpaca.close_position(ticker)
            except Exception as e:
                print(f"⚠️ Errore chiusura Alpaca {ticker}: {e}")
        
        # Reset stato
        state['entry'] = 0.0
        state['direction'] = 0
        state['qty'] = 0
        state['stop_loss'] = 0.0
        state['take_profit'] = 0.0
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

    def execute(self):
        """Logica principale della strategia intraday V7.0."""
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
            # Determina shape delle feature su un ticker campione
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
            model = get_model("7.0", self.model_path, shape_t=(LOOKBACK, n_tech), shape_m=(LOOKBACK, n_macro))
        
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
        
        for ticker in tqdm(TARGET_TICKERS_V70, desc="🎯 Analisi Intraday"):
            state = self._load_state(ticker)
            
            # Carica dati intraday aggiornati (1 minuto per SL/TP)
            df_raw = DataManager.get_cached_intraday_data(
                ticker, self.conn_market, ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
            )
            
            if len(df_raw) < LOOKBACK * 5 + 100:
                continue
            
            # Prezzo corrente dall'ultimo dato a 1 minuto (massima precisione per SL/TP)
            prezzo_corrente = df_raw['Close'].iloc[-1]
            
            # ── Gestione posizioni aperte (monitoraggio a 1 minuto) ──
            if state['direction'] != 0:
                rend_pct = ((prezzo_corrente - state['entry']) / state['entry']) * state['direction']
                
                # Stop Loss
                if rend_pct <= -STOP_LOSS_PCT:
                    msg = self._close_position(ticker, state, prezzo_corrente, 'SL')
                    logs_chiusure.append(msg)
                    continue
                
                # Take Profit
                if rend_pct >= TAKE_PROFIT_PCT:
                    msg = self._close_position(ticker, state, prezzo_corrente, 'TP')
                    logs_chiusure.append(msg)
                    continue
                
                # Chiusura EOD
                if close_only:
                    msg = self._close_position(ticker, state, prezzo_corrente, 'EOD')
                    logs_chiusure.append(msg)
                    continue
            
            # ── Generazione segnali (solo ogni 5 min, non close_only, non in posizione) ──
            if close_only or state['direction'] != 0 or not is_5min or model is None:
                continue
            
            # Resample a 5 minuti per il modello
            df_5min = DataManager.resample_to_5min(df_raw)
            df_feat = FeatureEngine.process_intraday_features(df_5min, self.macro)
            
            if len(df_feat) < LOOKBACK + 1:
                continue
            
            t_feat, m_feat = FeatureEngine.extract_intraday_features(df_feat, macro_labels=MACRO_LABELS_ORDERED)
            
            # Normalizza e prepara input
            scaler_t = StandardScaler().fit(t_feat)
            scaler_m = StandardScaler().fit(m_feat)
            
            X_t = np.array([scaler_t.transform(t_feat)[-LOOKBACK:]], dtype=np.float32)
            X_m = np.array([scaler_m.transform(m_feat)[-LOOKBACK:]], dtype=np.float32)
            
            if X_t.shape[1] < LOOKBACK:
                continue
            
            # Previsione
            pred = model.predict([X_t, X_m], verbose=0)[0][0]
            delta = pred - 0.50
            
            if abs(delta) > SIGNAL_THRESHOLD:
                direction = 1 if delta > 0 else -1
                side = OrderSide.BUY if direction == 1 else OrderSide.SELL
                side_label = "BUY LONG" if direction == 1 else "SELL SHORT"
                
                # Calcolo dimensione posizione
                budget = min(portfolio_value * MAX_ALLOCATION, cash_disponibile * 0.95)
                qty = int(budget // prezzo_corrente)
                
                if qty > 0 and budget > 100:
                    try:
                        if alpaca:
                            alpaca.submit_order(MarketOrderRequest(
                                symbol=ticker, qty=qty,
                                side=side, time_in_force=TimeInForce.DAY
                            ))
                        
                        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        sl = prezzo_corrente * (1 - STOP_LOSS_PCT * direction)
                        tp = prezzo_corrente * (1 + TAKE_PROFIT_PCT * direction)
                        investimento = qty * prezzo_corrente
                        
                        state['entry'] = prezzo_corrente
                        state['direction'] = direction
                        state['qty'] = qty
                        state['stop_loss'] = sl
                        state['take_profit'] = tp
                        state['entry_time'] = now_str
                        state['invested'] = investimento
                        self._save_state(ticker, state)
                        
                        cash_disponibile -= investimento
                        
                        emoji = "🟢" if direction == 1 else "🔴"
                        conf_pct = abs(delta) * 200  # confidence %
                        operazioni_fatte.append(
                            f"{emoji} <b>{ticker}</b>: {side_label} ${int(investimento)} "
                            f"(conf: {conf_pct:.0f}%)"
                        )
                    except Exception as e:
                        print(f"❌ Errore ordine {ticker}: {e}")
        
        # ── Report ──
        # Calcola statistiche dal DB storico
        win_rate_str = ""
        try:
            total = self.conn_trades.execute("SELECT count(*) FROM trades_history_v70").fetchone()[0]
            wins = self.conn_trades.execute("SELECT count(*) FROM trades_history_v70 WHERE pnl > 0").fetchone()[0]
            if total > 0:
                win_rate_str = f"{wins}/{total} ({wins/total*100:.1f}%)"
        except:
            pass
        
        trades_str = "\n".join(operazioni_fatte) if operazioni_fatte else ""
        logs_str = "\n".join(logs_chiusure) if logs_chiusure else ""
        
        # Stato posizioni aperte
        pos_aperte = sum(1 for t in TARGET_TICKERS_V70 if self._load_state(t)['direction'] != 0)
        
        mode = 'SOLO CHIUSURE' if close_only else ('SEGNALI 5min' if is_5min else 'MONITORAGGIO SL/TP')
        
        self.set_report_data(
            balance_str=f"${portfolio_value:,.2f}",
            win_rate_str=win_rate_str,
            trades_str=trades_str,
            logs_str=logs_str,
            extra_str=f"Posizioni aperte: {pos_aperte}/{len(TARGET_TICKERS_V70)} | "
                      f"Modalità: {mode}"
        )


def run():
    StrategyV70().run()


if __name__ == "__main__":
    run()
