"""
sync_market_data.py — Aggiornamento periodico dei dati di mercato.

Scarica i dati OHLCV più recenti da Yahoo Finance per tutti i ticker
(azioni, macro e crypto) e li salva nel database condiviso market_data.db.

Uso:
    python sync_market_data.py                  # Aggiorna solo i ticker del progetto (dal 2020)
    python sync_market_data.py --full           # Scarica TUTTO S&P 500 dal 1990
    python sync_market_data.py --crypto         # Includi anche crypto
    python sync_market_data.py --full --crypto  # Tutto: S&P 500 + crypto dal 1990
    python sync_market_data.py -v               # Output dettagliato
"""

import os
import sys
import argparse
import datetime
import time
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.config import (
    DB_MARKET, TARGET_TICKERS_AZIONARIO, TARGET_TICKERS_V43,
    BASE_TICKERS_V43, TARGET_TICKERS_CRIPTO, MACRO_MAP
)
from core.data_manager import DataManager

# ── S&P 500 Components (aggiornati a Marzo 2026) ─────────
SP500_TICKERS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE',
    'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK',
    'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN',
    'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE',
    'ARES', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO',
    'BA', 'BAC', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIO', 'BIIB',
    'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA',
    'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI',
    'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW',
    'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI',
    'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COO', 'COP', 'COST', 'CPB', 'CPRT',
    'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT',
    'CTRA', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'CZR',
    'D', 'DAL', 'DASH', 'DD', 'DE', 'DECK', 'DELL', 'DFS', 'DG', 'DGX',
    'DHI', 'DHR', 'DIS', 'DISH', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI',
    'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EME', 'EMN', 'EMR',
    'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN',
    'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FBHS', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS',
    'FISV', 'FITB', 'FIX', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRT', 'FTNT', 'FTV',
    'GD', 'GDDY', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG',
    'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
    'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HOLX', 'HON', 'HOOD', 'HPE', 'HPQ',
    'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM',
    'IBM', 'IBKR', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU',
    'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
    'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JPM',
    'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KMX',
    'KO', 'KR',
    'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LITE', 'LKQ', 'LLY', 'LMT',
    'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV',
    'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT',
    'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST',
    'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI',
    'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
    'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW',
    'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA',
    'NXPI',
    'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY',
    'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG',
    'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PLTR', 'PM', 'PNC',
    'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH',
    'PWR', 'PXD', 'PYPL',
    'QCOM', 'QRVO',
    'RCL', 'RE', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL',
    'ROP', 'ROST', 'RSG', 'RTX',
    'SATS', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SJM', 'SLB', 'SNA',
    'SMCI', 'SNPS', 'SO', 'SPG', 'SPGI', 'SQ', 'SRE', 'STE', 'STT', 'STX',
    'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY',
    'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT',
    'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV',
    'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL',
    'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
    'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRT', 'VRTX',
    'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM',
    'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN',
    'XEL', 'XOM', 'XRAY', 'XYL',
    'YUM',
    'ZBH', 'ZBRA', 'ZION', 'ZTS'
]


def sync(include_crypto=False, full_history=False, verbose=False):
    start_date = "1990-01-01" if full_history else "2020-01-01"
    mode_label = "COMPLETO (S&P 500 dal 1990)" if full_history else "STANDARD"

    print("=" * 60)
    print(f"📡 SYNC DATI DI MERCATO — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Modalità: {mode_label}")
    print("=" * 60)

    conn = DataManager.setup_db(DB_MARKET)

    # ── Costruisci la lista completa di ticker da sincronizzare ──
    all_tickers = set()

    if full_history:
        # S&P 500 completo
        all_tickers.update(SP500_TICKERS)
    
    # Ticker azionari del progetto (sempre inclusi)
    all_tickers.update(TARGET_TICKERS_AZIONARIO)
    all_tickers.update(TARGET_TICKERS_V43)
    all_tickers.update(BASE_TICKERS_V43)

    # Ticker macro (sempre inclusi)
    all_tickers.update(MACRO_MAP.keys())

    # Crypto (opzionale)
    if include_crypto:
        all_tickers.update(TARGET_TICKERS_CRIPTO)

    all_tickers = sorted(all_tickers)

    print(f"\n📊 Ticker da sincronizzare: {len(all_tickers)}")
    print(f"📅 Start date: {start_date}")
    if include_crypto:
        print("   (incluse crypto)")
    print(f"📂 Database: {DB_MARKET}")
    print()

    successi = 0
    errori = 0
    nuovi_dati = 0
    errori_lista = []

    for i, ticker in enumerate(all_tickers, 1):
        try:
            # Conta righe prima
            count_before = conn.execute(
                "SELECT count(*) FROM market_cache WHERE Ticker=?", (ticker,)
            ).fetchone()[0] if conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_cache'"
            ).fetchone() else 0

            # Scarica / aggiorna con la start_date corretta
            DataManager.get_cached_market_data(ticker, conn, start_date=start_date)

            # Conta righe dopo
            count_after = conn.execute(
                "SELECT count(*) FROM market_cache WHERE Ticker=?", (ticker,)
            ).fetchone()[0]

            nuove = count_after - count_before
            nuovi_dati += nuove
            successi += 1

            if verbose or nuove > 0:
                status = f"+{nuove} righe" if nuove > 0 else "aggiornato"
                print(f"  [{i:3d}/{len(all_tickers)}] ✅ {ticker:10s} — {status} (totale: {count_after})")

        except Exception as e:
            errori += 1
            errori_lista.append(ticker)
            if verbose:
                print(f"  [{i:3d}/{len(all_tickers)}] ❌ {ticker:10s} — Errore: {e}")

    conn.close()

    # ── Report finale ──
    print()
    print("=" * 60)
    print(f"✅ SYNC COMPLETATA")
    print(f"   Ticker elaborati: {successi}/{len(all_tickers)}")
    print(f"   Nuove righe inserite: {nuovi_dati}")
    if errori > 0:
        print(f"   ⚠️  Errori ({errori}): {', '.join(errori_lista[:10])}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sincronizza dati di mercato nel DB condiviso")
    parser.add_argument("--full", action="store_true", help="Scarica TUTTO S&P 500 dal 1990 (altrimenti solo ticker del progetto dal 2020)")
    parser.add_argument("--crypto", action="store_true", help="Includi anche i ticker crypto")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output dettagliato per ogni ticker")
    args = parser.parse_args()

    sync(include_crypto=args.crypto, full_history=args.full, verbose=args.verbose)
