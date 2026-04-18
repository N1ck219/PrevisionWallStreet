"""
download_intraday_data.py — Download massivo di dati a 1 minuto da Alpaca Markets.

Scarica e salva nel database condiviso (market_data.db) lo storico delle barre
a 1 minuto per tutti i ticker della strategia V7.0.

Uso:
    python download_intraday_data.py                # Download tutti i ticker V7.0
    python download_intraday_data.py --ticker AAPL   # Download singolo ticker
    python download_intraday_data.py --status        # Mostra stato download
    python download_intraday_data.py --start 2020-01-01  # Data inizio personalizzata
"""

import os
import sys
import argparse
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from core.config import (
    DB_MARKET, DB_MARKET_V70, TARGET_TICKERS_V70, MACRO_MAP,
    ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7
)
from core.data.data_manager import DataManager
from tqdm import tqdm


def show_status(conn):
    """Mostra lo stato attuale del download dei dati intraday."""
    print("\n📊 STATO DOWNLOAD DATI INTRADAY (1-min)")
    print("=" * 60)

    try:
        # ...
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='intraday_cache'"
        ).fetchone()

        if not exists:
            print("⚠️  Tabella intraday_cache non ancora creata.")
            print("   Esegui il download per iniziare.")
            return

        tickers = conn.execute(
            "SELECT DISTINCT Ticker FROM intraday_cache ORDER BY Ticker"
        ).fetchall()

        if not tickers:
            print("⚠️  Nessun dato presente. Esegui il download.")
            return

        total_rows = 0
        for (ticker,) in tickers:
            count = conn.execute(
                "SELECT count(*) FROM intraday_cache WHERE Ticker=?", (ticker,)
            ).fetchone()[0]
            min_dt = conn.execute(
                "SELECT MIN(Datetime) FROM intraday_cache WHERE Ticker=?", (ticker,)
            ).fetchone()[0]
            max_dt = conn.execute(
                "SELECT MAX(Datetime) FROM intraday_cache WHERE Ticker=?", (ticker,)
            ).fetchone()[0]
            total_rows += count
            
            in_v70 = "✅" if ticker in TARGET_TICKERS_V70 else "  "
            print(f"  {in_v70} {ticker:8s} — {count:>10,} barre | {min_dt} → {max_dt}")

        print(f"\n  📦 Totale: {total_rows:,} barre | {len(tickers)} ticker")

    except Exception as e:
        print(f"❌ Errore: {e}")

    print("=" * 60)


def download_all(tickers, api_key, secret_key, start_date, conn):
    """Scarica dati intraday per tutti i ticker specificati."""
    print(f"\n📡 DOWNLOAD DATI INTRADAY — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Ticker: {len(tickers)}")
    print(f"   Start:  {start_date}")
    print(f"   DB:     {DB_MARKET_V70}")
    print("=" * 60)

    for ticker in tqdm(tickers, desc="Download Ticker"):
        try:
            count_before = 0
            try:
                count_before = conn.execute(
                    "SELECT count(*) FROM intraday_cache WHERE Ticker=?", (ticker,)
                ).fetchone()[0]
            except:
                pass

            DataManager.get_cached_intraday_data(
                ticker, conn, api_key, secret_key, start_date=start_date
            )

            count_after = conn.execute(
                "SELECT count(*) FROM intraday_cache WHERE Ticker=?", (ticker,)
            ).fetchone()[0]

            nuove = count_after - count_before
            if nuove > 0:
                print(f"  ✅ {ticker:8s} — +{nuove:,} barre (totale: {count_after:,})")
            else:
                print(f"  ✓  {ticker:8s} — aggiornato ({count_after:,} barre)")

        except Exception as e:
            print(f"  ❌ {ticker:8s} — Errore: {e}")

    print("\n✅ DOWNLOAD COMPLETATO!")
    show_status(conn)


def main():
    parser = argparse.ArgumentParser(description="Download dati intraday 1-min da Alpaca Markets")
    parser.add_argument("--ticker", type=str, help="Download singolo ticker")
    parser.add_argument("--status", action="store_true", help="Mostra stato download")
    parser.add_argument("--start", type=str, default="2019-01-01",
                        help="Data inizio download (default: 2019-01-01)")
    parser.add_argument("--include-macro", action="store_true",
                        help="Includi anche i ticker macro (QQQ, GLD, SOXX)")
    args = parser.parse_args()

    if not ALPACA_API_KEY_7 or not ALPACA_SECRET_KEY_7:
        print("❌ ERRORE: Variabili ALPACA_API_KEY_7 e ALPACA_SECRET_KEY_7 non configurate nel .env!")
        sys.exit(1)

    conn = DataManager.setup_db(DB_MARKET_V70)

    if args.status:
        show_status(conn)
        conn.close()
        return

    tickers = [args.ticker] if args.ticker else list(TARGET_TICKERS_V70)
    
    if args.include_macro:
        # Aggiungi ticker macro che hanno dati intraday su Alpaca (no ^VIX, ^TNX)
        macro_intraday = ['QQQ', 'SOXX', 'GLD']
        tickers.extend([t for t in macro_intraday if t not in tickers])

    download_all(tickers, ALPACA_API_KEY_7, ALPACA_SECRET_KEY_7, args.start, conn)
    conn.close()


if __name__ == "__main__":
    main()
