"""
migrate_databases.py — Script di migrazione una tantum.

Sposta i dati dai vecchi database monolitici alla nuova struttura separata:
  - market_cache           → market_data.db   (condiviso)
  - portfolio/history V4.3 → trades_v4_3.db
  - portfolio/history V4.6 → trades_v4_6.db
  - state V5.6             → trades_v5_6.db
  - state V6.4             → trades_v6_4.db

Uso:
    python migrate_databases.py
"""

import os
import sys
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.config import DATA_DIR, DB_MARKET, DB_STOCK_V4, DB_STOCK_V45, DB_TRADES_V43, DB_TRADES_V46, DB_TRADES_V56, DB_TRADES_V64


def copy_table(src_conn, dst_conn, table_name, create_sql=None):
    """Copia una tabella da un DB sorgente a un DB destinazione."""
    # Verifica che la tabella esista nel sorgente
    exists = src_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    if not exists:
        print(f"  ⏭️  Tabella '{table_name}' non trovata nel sorgente, skip.")
        return 0

    # Leggi lo schema dal sorgente se non fornito
    if not create_sql:
        schema_row = src_conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone()
        if not schema_row or not schema_row[0]:
            print(f"  ⚠️  Schema non leggibile per '{table_name}', skip.")
            return 0
        create_sql = schema_row[0]

    # Crea la tabella nel destinazione (se non esiste)
    create_sql_safe = create_sql.replace(f"CREATE TABLE {table_name}", f"CREATE TABLE IF NOT EXISTS {table_name}")
    create_sql_safe = create_sql_safe.replace(f"CREATE TABLE \"{table_name}\"", f"CREATE TABLE IF NOT EXISTS \"{table_name}\"")
    dst_conn.execute(create_sql_safe)

    # Conta righe già presenti nella destinazione
    existing = dst_conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    if existing > 0:
        print(f"  ⚠️  Tabella '{table_name}' già contiene {existing} righe nel destinazione, skip per evitare duplicati.")
        return existing

    # Copia i dati
    rows = src_conn.execute(f"SELECT * FROM {table_name}").fetchall()
    if not rows:
        print(f"  ℹ️  Tabella '{table_name}' vuota nel sorgente.")
        return 0

    cols = len(rows[0])
    placeholders = ", ".join(["?"] * cols)
    dst_conn.executemany(f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})", rows)
    dst_conn.commit()
    print(f"  ✅ '{table_name}': {len(rows)} righe copiate.")
    return len(rows)


def migrate():
    print("=" * 60)
    print("🔄 MIGRAZIONE DATABASE — PrevisionWallStreet")
    print("=" * 60)

    total_rows = 0

    # ──────────────────────────────────────────────────────────
    # 1. market_cache → market_data.db (da entrambi i vecchi DB)
    # ──────────────────────────────────────────────────────────
    print("\n📦 [1/5] Migrazione dati di mercato → market_data.db")
    dst_market = sqlite3.connect(DB_MARKET)

    for old_db_path, label in [(DB_STOCK_V4, "stock_data.db"), (DB_STOCK_V45, "stock_data_v45.db")]:
        if not os.path.exists(old_db_path):
            print(f"  ⏭️  {label} non trovato, skip.")
            continue
        print(f"  📂 Lettura da {label}...")
        src = sqlite3.connect(old_db_path)
        total_rows += copy_table(src, dst_market, "market_cache")
        src.close()

    dst_market.close()

    # ──────────────────────────────────────────────────────────
    # 2. V4.3 trades → trades_v4_3.db
    # ──────────────────────────────────────────────────────────
    print("\n📦 [2/5] Migrazione operazioni V4.3 → trades_v4_3.db")
    if os.path.exists(DB_STOCK_V4):
        src = sqlite3.connect(DB_STOCK_V4)
        dst = sqlite3.connect(DB_TRADES_V43)
        for table in ["portfolio_v42", "history_v42", "portfolio_v43", "history_v43", "benchmark_bh"]:
            total_rows += copy_table(src, dst, table)
        src.close()
        dst.close()
    else:
        print("  ⏭️  stock_data.db non trovato, skip.")

    # ──────────────────────────────────────────────────────────
    # 3. V4.6 trades → trades_v4_6.db
    # ──────────────────────────────────────────────────────────
    print("\n📦 [3/5] Migrazione operazioni V4.6 → trades_v4_6.db")
    if os.path.exists(DB_STOCK_V45):
        src = sqlite3.connect(DB_STOCK_V45)
        dst = sqlite3.connect(DB_TRADES_V46)
        for table in ["portfolio_live_46", "history_live_46"]:
            total_rows += copy_table(src, dst, table)
        src.close()
        dst.close()
    else:
        print("  ⏭️  stock_data_v45.db non trovato, skip.")

    # ──────────────────────────────────────────────────────────
    # 4. V5.6 state → trades_v5_6.db
    # ──────────────────────────────────────────────────────────
    print("\n📦 [4/5] Migrazione stato V5.6 → trades_v5_6.db")
    if os.path.exists(DB_STOCK_V45):
        src = sqlite3.connect(DB_STOCK_V45)
        dst = sqlite3.connect(DB_TRADES_V56)
        total_rows += copy_table(src, dst, "state_v56")
        src.close()
        dst.close()
    else:
        print("  ⏭️  stock_data_v45.db non trovato, skip.")

    # ──────────────────────────────────────────────────────────
    # 5. V6.4 state → trades_v6_4.db
    # ──────────────────────────────────────────────────────────
    print("\n📦 [5/5] Migrazione stato V6.4 → trades_v6_4.db")
    if os.path.exists(DB_STOCK_V45):
        src = sqlite3.connect(DB_STOCK_V45)
        dst = sqlite3.connect(DB_TRADES_V64)
        total_rows += copy_table(src, dst, "state_v64")
        src.close()
        dst.close()
    else:
        print("  ⏭️  stock_data_v45.db non trovato, skip.")

    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"✅ MIGRAZIONE COMPLETATA — {total_rows} righe totali elaborate.")
    print("=" * 60)
    print("\nℹ️  I vecchi file DB NON sono stati eliminati.")
    print("   Puoi rimuoverli manualmente dopo aver verificato che tutto funziona:")
    print(f"   - {DB_STOCK_V4}")
    print(f"   - {DB_STOCK_V45}")


if __name__ == "__main__":
    migrate()
