"""
backtest_v7_3.py — Backtest intraday del modello Split Brain V7.3.

Simulazione rigorosa sugli ultimi 500 giorni di borsa con logica intraday:
- Dati a 1 minuto resampleati a 5 minuti per il modello
- Previsione a 1 ora (12 barre a 5 min)
- Sizing Dinamico in base alla forza del segnale
- Stop loss 2%, Take profit 1.5%
- Chiusura forzata a fine sessione (no overnight)
- Report con equity curve, win rate, Sharpe ratio, max drawdown

Uso:
    python simulations/backtest_v7_3.py
    python simulations/backtest_v7_3.py --days 250
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

REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'v7_3')
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "intraday_brain_v7_0.h5")

LOOKBACK = 60           # Finestra lookback (60 barre a 5-min = 5 ore)
HORIZON = 12            # Orizzonte previsione (12 barre a 5-min = 1 ora)
TEST_DAYS = 500         # Ultimi 500 giorni di borsa
CAPITALE_INIZIALE = 100_000.0
MAX_ALLOCATION = 0.20   # Max 20% su singolo asset
SIGNAL_THRESHOLD = 0.06 # Soglia minima per entrare
MAX_TRADES_DAY = 5      # Max trade per giorno per asset

# --- V7.3 SIMULAZIONE REALE ---
SLIPPAGE_PCT = 0.00015  # 0.015% per trade (in / out)
SEC_FEE_RATE = 0.0000229 # $22.90 per milione di controvalore venduto
FINRA_TAF_RATE = 0.000195 # per azione venduta (max $9.79)
MAX_DAILY_LOSS_PCT = -0.015 # Ferma i trade se si perde oltre l'1.5% in un giorno

def calculate_exit_fees(sell_price, qty):
    """Calcola le fee regolamentari applicabili (SEC e FINRA TAF) sul lato vendita."""
    principal = sell_price * qty
    sec_fee = principal * SEC_FEE_RATE
    taf_fee = min(qty * FINRA_TAF_RATE, 9.79)
    return sec_fee + taf_fee


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
    """Esegue il backtest intraday V7.3 su barre a 5 minuti."""
    print(f"\n🚀 AVVIO BACKTEST V7.3 (DYNAMIC SIZING & PRO REAL-WORLD) — {test_days} Giorni")
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
        atr = df_full['ATRr_14'].values if 'ATRr_14' in df_full.columns else np.zeros(len(df_full))
        
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
            'preds': preds,
            'atr': atr
        }
        print(f"  ✅ {ticker}: Pre-elaborazione completata ({len(valid_indices)} inferenze batch).")
        
        # Pulizia memoria esplicita profonda
        import gc
        del X_t_list, X_m_list, valid_indices
        if 'X_t_raw' in locals(): del X_t_raw
        if 'X_m_raw' in locals(): del X_m_raw
        gc.collect()
        
    # ── SIMULAZIONE TIME-SEQUENTIAL ──
    cash = CAPITALE_INIZIALE
    storia_equity = []
    tutti_trade = []
    trade_per_ora = {h: 0 for h in range(9, 17)}
    
    print(f"\n🏁 Inizio simulazione Time-Sequential ({len(test_dates)} giorni su barre a 5 min)...")
    
    for data_odierna in tqdm(test_dates, desc="Simulazione Giorni"):
        cash_inizio_giorno = cash
        giornata_sospesa = False
        
        # Estrazione e indicizzazione rapida dei dati di oggi per tutti i ticker
        giorno_data = {}
        time_slots_set = set()
        
        for ticker in dati_ticker.keys():
            opt = ticker_data_optimized[ticker]
            mask = opt['date_array'] == data_odierna
            indices = np.where(mask)[0]
            if len(indices) < 10: 
                continue
            
            time_map = {}
            for idx in indices:
                if idx < LOOKBACK or idx + HORIZON >= len(opt['prezzi']):
                    continue
                h, m = opt['ore'][idx], opt['minuti'][idx]
                time_map[(h, m)] = idx
                time_slots_set.add((h, m))
                
            if time_map:
                giorno_data[ticker] = {
                    'time_map': time_map,
                    'opt': opt,
                    'trades_today': 0
                }
                
        if not giorno_data:
            storia_equity.append(cash)
            continue
            
        time_slots = sorted(list(time_slots_set)) # [ (9,30), (9,35), ..., (16,0) ]
        posizioni_attive = {} # ticker -> pos
        
        for h, m in time_slots:
            is_eod = (h >= 15 and m >= 30)
            
            # --- 1. GESTIONE POSIZIONI APERTE ---
            tickers_to_close = []
            for ticker, pos in posizioni_attive.items():
                if ticker not in giorno_data or (h, m) not in giorno_data[ticker]['time_map']:
                    continue
                
                bar_idx = giorno_data[ticker]['time_map'][(h, m)]
                prezzo = giorno_data[ticker]['opt']['prezzi'][bar_idx]
                pos['last_price'] = prezzo
                
                # Equity corrente include cash + valore flottante delle posizioni aperte
                valore_posizioni = 0
                for p in posizioni_attive.values():
                    valore_posizioni += p['qty'] * p['entry'] + (p['last_price'] - p['entry']) * p['qty'] * p['direction']
                
                equity_corrente = cash + valore_posizioni
                
                # Controllo Daily Loss Limit (se l'equity flottante giornaliera crolla)
                current_day_pnl_pct = (equity_corrente - cash_inizio_giorno) / cash_inizio_giorno
                if not is_eod and current_day_pnl_pct <= MAX_DAILY_LOSS_PCT and not giornata_sospesa:
                    giornata_sospesa = True 
                    
                if giornata_sospesa and not is_eod:
                    is_eod = True # Alza il flag EOD forzosamente per liquidare le posizioni a mercato
                
                if is_eod:
                    # Chiusura fine giornata o Stop di Emergenza
                    exit_price = prezzo * (1 - SLIPPAGE_PCT) if pos['direction'] == 1 else prezzo * (1 + SLIPPAGE_PCT)
                    fees = calculate_exit_fees(exit_price, pos['qty']) if pos['direction'] == 1 else calculate_exit_fees(pos['entry'], pos['qty'])
                    
                    pnl = (exit_price - pos['entry']) * pos['qty'] * pos['direction'] - fees
                    cash += pos['qty'] * pos['entry'] + pnl
                    
                    tutti_trade.append({
                        'ticker': ticker, 'date': str(data_odierna),
                        'direction': pos['direction'], 'entry': pos['entry'], 'exit': exit_price,
                        'pnl': pnl, 'reason': 'STOP_GG' if giornata_sospesa else 'EOD', 'hour': h
                    })
                    if h in trade_per_ora: trade_per_ora[h] += 1
                    tickers_to_close.append(ticker)
                    continue
                    
                # SMART EXIT LOGIC
                rend = ((prezzo - pos['entry']) / pos['entry']) * pos['direction']
                if pos['direction'] == 1:
                    if prezzo > pos['highest']: pos['highest'] = prezzo
                    max_rend = (pos['highest'] - pos['entry']) / pos['entry']
                else:
                    if prezzo < pos['lowest']: pos['lowest'] = prezzo
                    max_rend = (pos['entry'] - pos['lowest']) / pos['entry']
                    
                if pos['status'] == 'normal' and max_rend >= 0.01:
                    pos['status'] = 'break_even'
                if pos['status'] in ['normal', 'break_even'] and max_rend >= 0.015:
                    pos['status'] = 'trailing'
                    
                if pos['status'] == 'normal':
                    sl_price = pos['entry'] * (1 - pos['sl'] * pos['direction'])
                elif pos['status'] == 'break_even':
                    sl_price = pos['entry'] * (1 + 0.001 * pos['direction'])
                else:
                    sl_price = pos['highest'] * (1 - 0.008) if pos['direction'] == 1 else pos['lowest'] * (1 + 0.008)
                    
                is_sl = (pos['direction'] == 1 and prezzo <= sl_price) or (pos['direction'] == -1 and prezzo >= sl_price)
                
                if is_sl:
                    exit_price = prezzo * (1 - SLIPPAGE_PCT) if pos['direction'] == 1 else prezzo * (1 + SLIPPAGE_PCT)
                    fees = calculate_exit_fees(exit_price, pos['qty']) if pos['direction'] == 1 else calculate_exit_fees(pos['entry'], pos['qty'])

                    pnl = (exit_price - pos['entry']) * pos['qty'] * pos['direction'] - fees
                    cash += pos['qty'] * pos['entry'] + pnl
                    reason = 'TRAIL' if pos['status'] == 'trailing' else ('BE' if pos['status'] == 'break_even' else 'SL')
                    tutti_trade.append({
                        'ticker': ticker, 'date': str(data_odierna),
                        'direction': pos['direction'], 'entry': pos['entry'], 'exit': exit_price,
                        'pnl': pnl, 'reason': reason, 'hour': h
                    })
                    if h in trade_per_ora: trade_per_ora[h] += 1
                    tickers_to_close.append(ticker)
                    continue
                
                # Take Profit Parziale
                if pos['stage'] == 1 and rend >= pos['tp']:
                    sell_qty = pos['qty'] // 2
                    if sell_qty > 0:
                        exit_price = prezzo * (1 - SLIPPAGE_PCT) if pos['direction'] == 1 else prezzo * (1 + SLIPPAGE_PCT)
                        fees = calculate_exit_fees(exit_price, sell_qty) if pos['direction'] == 1 else calculate_exit_fees(pos['entry'], sell_qty)
                        
                        pnl = (exit_price - pos['entry']) * sell_qty * pos['direction'] - fees
                        cash += sell_qty * pos['entry'] + pnl
                        
                        tutti_trade.append({
                            'ticker': ticker, 'date': str(data_odierna),
                            'direction': pos['direction'], 'entry': pos['entry'], 'exit': exit_price,
                            'pnl': pnl, 'reason': 'TP_50%', 'hour': h
                        })
                        if h in trade_per_ora: trade_per_ora[h] += 1
                        
                        pos['qty'] -= sell_qty
                        pos['stage'] = 2
                        pos['status'] = 'trailing' # Il rimanente 50% è ora a briglia sciolta (Trailing)
                    
            for t in tickers_to_close:
                del posizioni_attive[t]
                
            # --- 2. GENERAZIONE SEGNALI E NUOVI INGRESSI ---
            if h < 15 and not is_eod and not giornata_sospesa:
                segnali = []
                for ticker, tk_data in giorno_data.items():
                    if ticker in posizioni_attive: continue
                    if tk_data['trades_today'] >= MAX_TRADES_DAY: continue
                    if (h, m) not in tk_data['time_map']: continue
                    
                    bar_idx = tk_data['time_map'][(h, m)]
                    pred = tk_data['opt']['preds'][bar_idx]
                    delta = pred - 0.50
                    if abs(delta) > SIGNAL_THRESHOLD:
                        segnali.append({
                            'ticker': ticker, 'delta': delta,
                            'prezzo': tk_data['opt']['prezzi'][bar_idx],
                            'atr': tk_data['opt']['atr'][bar_idx]
                        })
                
                # Ordina i segnali per confidenza decrescente
                segnali.sort(key=lambda x: abs(x['delta']), reverse=True)
                
                for s in segnali:
                    direction = 1 if s['delta'] > 0 else -1
                    entry_price = s['prezzo'] * (1 + SLIPPAGE_PCT) if direction == 1 else s['prezzo'] * (1 - SLIPPAGE_PCT)
                    
                    # Sizing Dinamico in base alla forza del segnale
                    abs_delta = abs(s['delta'])
                    if abs_delta >= 0.12:
                        alloc = 0.20
                    elif abs_delta >= 0.08:
                        alloc = 0.15
                    else:
                        alloc = 0.10
                        
                    budget = min(cash * alloc, cash * 0.95)
                    qty = int(budget // entry_price)
                    
                    if qty > 0 and budget > 100:
                        baseline_sl = (2.0 * s['atr'] / entry_price) if s['atr'] > 0 else 0.02
                        dynamic_sl = max(0.005, min(baseline_sl, 0.035))
                        dynamic_tp = dynamic_sl * 1.5
                        
                        posizioni_attive[s['ticker']] = {
                            'direction': direction,
                            'entry': entry_price,
                            'qty': qty,
                            'sl': dynamic_sl,
                            'tp': dynamic_tp,
                            'highest': entry_price,
                            'lowest': entry_price,
                            'status': 'normal',
                            'last_price': entry_price,
                            'stage': 1
                        }
                        cash -= qty * entry_price
                        giorno_data[s['ticker']]['trades_today'] += 1
                        if h in trade_per_ora: trade_per_ora[h] += 1

        # Chiusura d'emergenza fine giornata su eventuali residui
        for ticker, pos in list(posizioni_attive.items()):
            exit_price = pos['last_price'] * (1 - SLIPPAGE_PCT) if pos['direction'] == 1 else pos['last_price'] * (1 + SLIPPAGE_PCT)
            fees = calculate_exit_fees(exit_price, pos['qty']) if pos['direction'] == 1 else calculate_exit_fees(pos['entry'], pos['qty'])
            
            pnl = (exit_price - pos['entry']) * pos['qty'] * pos['direction'] - fees
            cash += pos['qty'] * pos['entry'] + pnl
            
            tutti_trade.append({
                'ticker': ticker, 'date': str(data_odierna),
                'direction': pos['direction'], 'entry': pos['entry'], 'exit': exit_price,
                'pnl': pnl, 'reason': 'STOP_GG' if giornata_sospesa else 'EOD', 'hour': 16
            })
        posizioni_attive.clear()

        # Checkpoint Equity fine giornata
        storia_equity.append(cash)

    conn_daily.close()
    conn_v70.close()
    
    # ── REPORT ──
    generate_report(storia_equity, tutti_trade, trade_per_ora, test_dates)


def generate_report(storia_equity, tutti_trade, trade_per_ora, test_dates):
    """Genera il report completo del backtest."""
    print("\n" + "=" * 60)
    print("📊 REPORT BACKTEST V7.3 (DYNAMIC SIZING)")
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
    fig.suptitle('Backtest V7.3 (Dynamic Sizing) — PRO Real-World', fontsize=16, fontweight='bold')
    
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
    report_path = os.path.join(REPORT_DIR, "backtest_v7_3_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Report salvato in: {report_path}")
    
    # Salva anche i trade in CSV
    if n_trade > 0:
        csv_path = os.path.join(REPORT_DIR, "trades_v7_3.csv")
        pd.DataFrame(tutti_trade).to_csv(csv_path, index=False)
        print(f"📋 Trade log salvato in: {csv_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest intraday V7.0 (5-min)")
    parser.add_argument("--days", type=int, default=TEST_DAYS,
                        help=f"Numero di giorni di test (default: {TEST_DAYS})")
    args = parser.parse_args()
    
    run_backtest(test_days=args.days)
