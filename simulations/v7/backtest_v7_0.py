"""
backtest_v7_0.py — Backtest intraday del modello Split Brain V7.0.

Simulazione rigorosa sugli ultimi 500 giorni di borsa con logica intraday:
- Dati a 1 minuto resampleati a 5 minuti per il modello
- Previsione a 1 ora (12 barre a 5 min)
- Stop loss 2%, Take profit 1.5%
- Chiusura forzata a fine sessione (no overnight)
- Report con equity curve, win rate, Sharpe ratio, max drawdown

Uso:
    python simulations/backtest_v7_0.py
    python simulations/backtest_v7_0.py --days 250
"""

import sys
import os
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from core.config import (
    DB_MARKET, DB_MARKET_V70, MODELS_DIR, TARGET_TICKERS_V70,
    MACRO_MAP, MACRO_LABELS_ORDERED,
    ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
)
from core.data_manager import DataManager
from core.features import FeatureEngine
from core.model_factory import get_model

REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'v7_0')
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "intraday_brain_v7_0.h5")

LOOKBACK = 60           # Finestra lookback (60 barre a 5-min = 5 ore)
HORIZON = 12            # Orizzonte previsione (12 barre a 5-min = 1 ora)
TEST_DAYS = 500         # Ultimi 500 giorni di borsa
CAPITALE_INIZIALE = 100_000.0
MAX_ALLOCATION = 0.20   # Max 20% su singolo asset
STOP_LOSS_PCT = 0.02    # Stop loss 2% (stretto per intraday)
TAKE_PROFIT_PCT = 0.015 # Take profit 1.5%
SIGNAL_THRESHOLD = 0.06 # Soglia minima per entrare
MAX_TRADES_DAY = 5      # Max trade per giorno per asset


def load_and_resample_data(conn, ticker, macro):
    """Carica dati a 1 minuto dal DB, resampla a 5 minuti e processa feature."""
    df_raw = pd.read_sql_query(
        "SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker=? ORDER BY Datetime ASC",
        conn, params=(ticker,)
    )
    
    if len(df_raw) < (LOOKBACK + HORIZON) * 5 + 500:
        return None
    
    # Resample a 5 minuti
    df_5min = DataManager.resample_to_5min(df_raw)
    
    if len(df_5min) < LOOKBACK + HORIZON + 100:
        return None
    
    # Feature engineering sulle barre a 5 minuti
    df_feat = FeatureEngine.process_intraday_features(df_5min, macro)
    return df_feat


def run_backtest(test_days=TEST_DAYS):
    """Esegue il backtest intraday V7.0 su barre a 5 minuti."""
    print(f"\n🚀 AVVIO BACKTEST V7.0 (Intraday Sniper 5min) — {test_days} Giorni")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: Modello non trovato in {MODEL_PATH}")
        print("   Esegui prima: python train_v7_0.py")
        return
    
    conn_daily = DataManager.setup_db(DB_MARKET)
    conn_v70 = DataManager.setup_db(DB_MARKET_V70)
    
    # Verifica disponibilità dati intraday
    try:
        exists = conn_v70.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='intraday_cache'"
        ).fetchone()
        if not exists:
            print("❌ ERRORE: Tabella intraday_cache non trovata nel nuovo DB!")
            print("   Esegui prima: python download_intraday_data.py")
            conn_daily.close()
            conn_v70.close()
            return
    except:
        pass
    
    # Carica macro dal DB daily standard
    print("📥 Caricamento dati macro da DB daily...")
    macro = {
        label: DataManager.get_cached_market_data(m, conn_daily, start_date="2000-01-01")[['Date', 'Close']].rename(columns={'Close': label})
        for m, label in MACRO_MAP.items()
    }
    
    # Carica, resampla e processa dati per ogni ticker
    print("📊 Elaborazione dati intraday (1min → 5min)...")
    dati_ticker = {}
    
    for ticker in tqdm(TARGET_TICKERS_V70, desc="Caricamento ticker"):
        df_feat = load_and_resample_data(conn_v70, ticker, macro)
        if df_feat is not None and len(df_feat) > LOOKBACK + HORIZON:
            dati_ticker[ticker] = df_feat
            print(f"  ✅ {ticker}: {len(df_feat):,} barre a 5-min")
        else:
            print(f"  ⚠️ {ticker}: dati insufficienti, skip")
    
    if not dati_ticker:
        print("❌ Nessun ticker con dati sufficienti per il backtest!")
        conn_daily.close()
        conn_v70.close()
        return
    
    # Determina i giorni unici di trading disponibili
    all_dates = set()
    for ticker, df in dati_ticker.items():
        dates = pd.to_datetime(df['Datetime']).dt.date.unique()
        all_dates.update(dates)
    
    all_dates_sorted = sorted(all_dates)
    
    # Prendi gli ultimi TEST_DAYS giorni
    if len(all_dates_sorted) > test_days:
        test_dates = all_dates_sorted[-test_days:]
    else:
        test_dates = all_dates_sorted
        print(f"⚠️ Solo {len(test_dates)} giorni disponibili (richiesti: {test_days})")
    
    print(f"\n📅 Periodo test: {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} giorni)")
    
    # Prepara il modello
    sample_df = list(dati_ticker.values())[0]
    t_sample, m_sample = FeatureEngine.extract_intraday_features(sample_df, macro_labels=MACRO_LABELS_ORDERED)
    n_tech = t_sample.shape[1]
    n_macro = m_sample.shape[1]
    
    tf.keras.backend.clear_session()
    model = get_model("7.0", MODEL_PATH, shape_t=(LOOKBACK, n_tech), shape_m=(LOOKBACK, n_macro))
    
    # Prepara scalers globali (fit su tutti i dati storici)
    all_tech = np.concatenate([FeatureEngine.extract_intraday_features(df, macro_labels=MACRO_LABELS_ORDERED)[0] for df in dati_ticker.values()])
    all_macro = np.concatenate([FeatureEngine.extract_intraday_features(df, macro_labels=MACRO_LABELS_ORDERED)[1] for df in dati_ticker.values()])
    scaler_t = StandardScaler().fit(all_tech)
    scaler_m = StandardScaler().fit(all_macro)
    
    # ── PRE-ELABORAZIONE E BATCH PREDICTION ──
    print("\n🧠 Calcolo previsioni in batch e ottimizzazione dati (Attendere, è veloce!)...")
    
    ticker_data_optimized = {}
    
    for ticker, df_full in dati_ticker.items():
        # Feature processing e Normalizzazione
        t_feat, m_feat = FeatureEngine.extract_intraday_features(df_full, macro_labels=MACRO_LABELS_ORDERED)
        t_norm = scaler_t.transform(t_feat)
        m_norm = scaler_m.transform(m_feat)
        
        # Colonne essenziali
        dt_col = pd.to_datetime(df_full['Datetime'])
        date_array = dt_col.dt.date.values
        ore = dt_col.dt.hour.values
        minuti = dt_col.dt.minute.values
        prezzo_col = 'prezzo' if 'prezzo' in df_full.columns else 'Close'
        prezzi = df_full[prezzo_col].values
        
        n_bars = len(df_full)
        preds = np.ones(n_bars, dtype=np.float32) * 0.5
        
        valid_indices = []
        X_t_list, X_m_list = [], []
        
        # Calcoliamo previsioni per tutte le barre dal lookback all'orizzonte
        for i in range(LOOKBACK, n_bars - HORIZON):
            X_t_list.append(t_norm[i - LOOKBACK:i])
            X_m_list.append(m_norm[i - LOOKBACK:i])
            valid_indices.append(i)
            
        if valid_indices:
            print(f"  🧠 Calcolo inferenze per {ticker} ({len(valid_indices)} barre)...")
            X_t_raw = np.array(X_t_list, dtype=np.float32)
            X_m_raw = np.array(X_m_list, dtype=np.float32)
            
            # Per evitare crash di memoria (InternalError), dividiamo il ticker in chunk più piccoli
            chunk_size = 25000
            all_preds = []
            
            for start_idx in range(0, len(X_t_raw), chunk_size):
                end_idx = min(start_idx + chunk_size, len(X_t_raw))
                X_t_chunk = X_t_raw[start_idx:end_idx]
                X_m_chunk = X_m_raw[start_idx:end_idx]
                
                # Inferenza rapida sul chunk (batch size ultra ristretto per prevenire CuDNN leaks)
                chunk_preds = model.predict([X_t_chunk, X_m_chunk], batch_size=128, verbose=0).flatten()
                all_preds.append(chunk_preds)
                
            batch_preds = np.concatenate(all_preds)
            
            for k, idx in enumerate(valid_indices):
                preds[idx] = batch_preds[k]
                
        ticker_data_optimized[ticker] = {
            'date_array': date_array,
            'ore': ore,
            'minuti': minuti,
            'prezzi': prezzi,
            'preds': preds
        }
        print(f"  ✅ {ticker}: Pre-elaborazione completata ({len(valid_indices)} inferenze batch).")
        
        # Pulizia memoria esplicita profonda
        import gc
        del X_t_list, X_m_list, valid_indices
        if 'X_t_raw' in locals(): del X_t_raw
        if 'X_m_raw' in locals(): del X_m_raw
        gc.collect()
        
    # ── SIMULAZIONE ──
    cash = CAPITALE_INIZIALE
    storia_equity = []
    tutti_trade = []
    trade_per_ora = {h: 0 for h in range(9, 17)}
    
    print(f"\n🏁 Inizio simulazione ({len(test_dates)} giorni su barre a 5 min)...")
    
    for data_odierna in tqdm(test_dates, desc="Simulazione Giorni"):
        # Per ogni ticker, simula la giornata intraday
        posizioni_giorno = {}  # ticker → {direction, entry, qty, stop, take}
        pnl_giorno = 0.0
        
        for ticker in dati_ticker.keys():
            opt_data = ticker_data_optimized[ticker]
            
            # Maschera booleana rapida per trovare gli indici del giorno corrente
            mask_giorno = opt_data['date_array'] == data_odierna
            idx_giorno = np.where(mask_giorno)[0]
            
            if len(idx_giorno) < 10:
                continue
            
            prezzi = opt_data['prezzi']
            ore = opt_data['ore']
            minuti = opt_data['minuti']
            preds = opt_data['preds']
            
            ticker_trades = 0
            pos = None  # posizione attiva per questo ticker/giorno
            
            # Itera su ogni barra a 5 minuti del giorno
            for bar_idx in idx_giorno:
                if bar_idx < LOOKBACK or bar_idx + HORIZON >= len(prezzi):
                    continue
                
                prezzo_corrente = prezzi[bar_idx]
                ora_corrente = ore[bar_idx]
                min_corrente = minuti[bar_idx]
                
                # Chiusura forzata a fine sessione (dopo le 15:30)
                if pos is not None and ora_corrente >= 15 and min_corrente >= 30:
                    pnl = (prezzo_corrente - pos['entry']) * pos['qty'] * pos['direction']
                    pnl_giorno += pnl
                    cash += pos['qty'] * prezzo_corrente
                    tutti_trade.append({
                        'ticker': ticker, 'date': str(data_odierna),
                        'direction': pos['direction'],
                        'entry': pos['entry'], 'exit': prezzo_corrente,
                        'pnl': pnl, 'reason': 'EOD', 'hour': ora_corrente
                    })
                    if ora_corrente in trade_per_ora:
                        trade_per_ora[ora_corrente] += 1
                    pos = None
                    continue
                
                # Gestione posizione aperta
                if pos is not None:
                    rend = ((prezzo_corrente - pos['entry']) / pos['entry']) * pos['direction']
                    
                    # Stop Loss
                    if rend <= -STOP_LOSS_PCT:
                        pnl = (prezzo_corrente - pos['entry']) * pos['qty'] * pos['direction']
                        pnl_giorno += pnl
                        cash += pos['qty'] * prezzo_corrente
                        tutti_trade.append({
                            'ticker': ticker, 'date': str(data_odierna),
                            'direction': pos['direction'],
                            'entry': pos['entry'], 'exit': prezzo_corrente,
                            'pnl': pnl, 'reason': 'SL', 'hour': ora_corrente
                        })
                        if ora_corrente in trade_per_ora:
                            trade_per_ora[ora_corrente] += 1
                        pos = None
                        continue
                    
                    # Take Profit
                    if rend >= TAKE_PROFIT_PCT:
                        pnl = (prezzo_corrente - pos['entry']) * pos['qty'] * pos['direction']
                        pnl_giorno += pnl
                        cash += pos['qty'] * prezzo_corrente
                        tutti_trade.append({
                            'ticker': ticker, 'date': str(data_odierna),
                            'direction': pos['direction'],
                            'entry': pos['entry'], 'exit': prezzo_corrente,
                            'pnl': pnl, 'reason': 'TP', 'hour': ora_corrente
                        })
                        if ora_corrente in trade_per_ora:
                            trade_per_ora[ora_corrente] += 1
                        pos = None
                        continue
                
                # Generazione segnale (solo se non in posizione e prima delle 15:00)
                if pos is None and ora_corrente < 15 and ticker_trades < MAX_TRADES_DAY:
                    pred = preds[bar_idx]
                    delta = pred - 0.50
                    
                    if abs(delta) > SIGNAL_THRESHOLD:
                        direction = 1 if delta > 0 else -1
                        budget = min(cash * MAX_ALLOCATION, cash * 0.95)
                        qty = int(budget // prezzo_corrente)
                        
                        if qty > 0 and budget > 100:
                            pos = {
                                'direction': direction,
                                'entry': prezzo_corrente,
                                'qty': qty
                            }
                            cash -= qty * prezzo_corrente
                            ticker_trades += 1
                            if ora_corrente in trade_per_ora:
                                trade_per_ora[ora_corrente] += 1
            
            # Chiudi eventuali posizioni rimaste aperte
            if pos is not None:
                ultimo_prezzo = prezzi[idx_giorno[-1]]
                pnl = (ultimo_prezzo - pos['entry']) * pos['qty'] * pos['direction']
                pnl_giorno += pnl
                cash += pos['qty'] * ultimo_prezzo
                tutti_trade.append({
                    'ticker': ticker, 'date': str(data_odierna),
                    'direction': pos['direction'],
                    'entry': pos['entry'], 'exit': ultimo_prezzo,
                    'pnl': pnl, 'reason': 'EOD', 'hour': 16
                })
                pos = None
        
        # Equity di fine giornata
        cash += pnl_giorno
        storia_equity.append(cash)
    
    conn_daily.close()
    conn_v70.close()
    
    # ── REPORT ──
    generate_report(storia_equity, tutti_trade, trade_per_ora, test_dates)


def generate_report(storia_equity, tutti_trade, trade_per_ora, test_dates):
    """Genera il report completo del backtest."""
    print("\n" + "=" * 60)
    print("📊 REPORT BACKTEST V7.0 (5-min Resampled)")
    print("=" * 60)
    
    equity_finale = storia_equity[-1] if storia_equity else CAPITALE_INIZIALE
    rendimento = ((equity_finale - CAPITALE_INIZIALE) / CAPITALE_INIZIALE) * 100
    
    # Statistiche trade
    n_trade = len(tutti_trade)
    if n_trade > 0:
        trade_df = pd.DataFrame(tutti_trade)
        win_trades = trade_df[trade_df['pnl'] > 0]
        loss_trades = trade_df[trade_df['pnl'] <= 0]
        win_rate = len(win_trades) / n_trade * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        profit_factor = abs(win_trades['pnl'].sum() / (loss_trades['pnl'].sum() + 1e-9))
        
        # Per motivo di chiusura
        reasons = trade_df['reason'].value_counts()
    else:
        win_rate = 0
        avg_win = avg_loss = profit_factor = 0
        reasons = pd.Series()
    
    # Max Drawdown
    equity_arr = np.array(storia_equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / (peak + 1e-9)
    max_dd = drawdown.min() * 100
    
    # Sharpe Ratio (annualizzato sui rendimenti giornalieri)
    if len(storia_equity) > 1:
        daily_returns = np.diff(storia_equity) / np.array(storia_equity[:-1])
        sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0
    
    print(f"\n💰 Capitale Iniziale:  ${CAPITALE_INIZIALE:>12,.2f}")
    print(f"💰 Capitale Finale:    ${equity_finale:>12,.2f}")
    print(f"📈 Rendimento:          {rendimento:>+10.2f}%")
    print(f"📉 Max Drawdown:        {max_dd:>10.2f}%")
    print(f"📊 Sharpe Ratio:        {sharpe:>10.2f}")
    print(f"\n🔄 Totale Trade:        {n_trade:>10}")
    print(f"✅ Win Rate:            {win_rate:>9.1f}%")
    print(f"💵 Avg Win:             ${avg_win:>10,.2f}")
    print(f"💸 Avg Loss:            ${avg_loss:>10,.2f}")
    print(f"⚖️ Profit Factor:       {profit_factor:>10.2f}")
    
    if len(reasons) > 0:
        print(f"\n📋 Motivi chiusura:")
        for reason, count in reasons.items():
            print(f"   {reason:5s}: {count:>5}")
    
    # ── GRAFICI ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Backtest V7.0 — Intraday Sniper (5-min Bars)', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    ax1 = axes[0, 0]
    ax1.plot(test_dates[:len(storia_equity)], storia_equity, color='#00C853', linewidth=1.5, label='Portafoglio')
    ax1.axhline(CAPITALE_INIZIALE, color='red', linestyle='--', alpha=0.5, label='Capitale Iniziale')
    ax1.fill_between(test_dates[:len(storia_equity)], CAPITALE_INIZIALE, storia_equity, 
                     where=[e >= CAPITALE_INIZIALE for e in storia_equity], color='#00C85330')
    ax1.fill_between(test_dates[:len(storia_equity)], CAPITALE_INIZIALE, storia_equity,
                     where=[e < CAPITALE_INIZIALE for e in storia_equity], color='#FF174430')
    ax1.set_title(f'Equity Curve — Rendimento: {rendimento:+.2f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    ax2.fill_between(test_dates[:len(drawdown)], drawdown * 100, color='#FF1744', alpha=0.4)
    ax2.plot(test_dates[:len(drawdown)], drawdown * 100, color='#FF1744', linewidth=0.8)
    ax2.set_title(f'Drawdown — Max: {max_dd:.2f}%')
    ax2.set_ylabel('%')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribuzione PnL per trade
    ax3 = axes[1, 0]
    if n_trade > 0:
        pnls = [t['pnl'] for t in tutti_trade]
        colors = ['#00C853' if p > 0 else '#FF1744' for p in pnls]
        ax3.hist(pnls, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
        ax3.axvline(0, color='white', linestyle='--', linewidth=1)
    ax3.set_title(f'Distribuzione PnL ({n_trade} trade)')
    ax3.set_xlabel('PnL ($)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade per ora del giorno
    ax4 = axes[1, 1]
    ore = list(trade_per_ora.keys())
    conteggi = list(trade_per_ora.values())
    bars = ax4.bar(ore, conteggi, color='#7C4DFF', alpha=0.8, edgecolor='white')
    ax4.set_title('Distribuzione Trade per Ora')
    ax4.set_xlabel('Ora (ET)')
    ax4.set_ylabel('Numero Trade')
    ax4.set_xticks(ore)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    report_path = os.path.join(REPORT_DIR, "backtest_v7_0_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Report salvato in: {report_path}")
    
    # Salva anche i trade in CSV
    if n_trade > 0:
        csv_path = os.path.join(REPORT_DIR, "trades_v7_0.csv")
        pd.DataFrame(tutti_trade).to_csv(csv_path, index=False)
        print(f"📋 Trade log salvato in: {csv_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest intraday V7.0 (5-min)")
    parser.add_argument("--days", type=int, default=TEST_DAYS,
                        help=f"Numero di giorni di test (default: {TEST_DAYS})")
    args = parser.parse_args()
    
    run_backtest(test_days=args.days)
