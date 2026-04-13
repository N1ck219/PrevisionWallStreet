"""
main.py — Entry point unificato per tutte le strategie.

Uso:
    python main.py --all                    # Azioni + Crypto
    python main.py --stock                  # Solo azioni (V4.3 → V6.4)
    python main.py --crypto                 # Solo crypto
    python main.py --strategy v4.3 v6.4     # Solo strategie specifiche
    python main.py --list                   # Mostra strategie disponibili
"""

import os
import sys
import gc
import time
import logging
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "reports", "run_log.txt"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Forza l'output del terminale in UTF-8 se possibile (utile su Windows)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

# ── Registry delle strategie ─────────────────────────────
STRATEGIES = {
    'v4.3':   {'module': 'strategies.strategy_v4_3',        'class': 'StrategyV43',       'type': 'stock'},
    'v4.6':   {'module': 'strategies.strategy_v4_6',        'class': 'StrategyV46',       'type': 'stock'},
    'v5.6':   {'module': 'strategies.strategy_v5_6',        'class': 'StrategyV56',       'type': 'stock'},
    'v6.4':   {'module': 'strategies.strategy_v6_4',        'class': 'StrategyV64',       'type': 'stock'},
    'crypto': {'module': 'strategies.strategy_crypto_v1_7', 'class': 'StrategyCryptoV17', 'type': 'crypto'},
}


def run_strategy(key):
    """Importa ed esegue una singola strategia."""
    import importlib
    info = STRATEGIES[key]
    module = importlib.import_module(info['module'])
    strategy_class = getattr(module, info['class'])
    strategy_class().run()


def main():
    parser = argparse.ArgumentParser(description="PrevisionWallStreet — Trading Bot Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Esegui tutte le strategie (azioni + crypto)')
    group.add_argument('--stock', action='store_true', help='Esegui solo le strategie azionarie')
    group.add_argument('--crypto', action='store_true', help='Esegui solo la strategia crypto')
    group.add_argument('--strategy', nargs='+', choices=STRATEGIES.keys(), help='Esegui strategie specifiche')
    group.add_argument('--list', action='store_true', help='Mostra le strategie disponibili')
    parser.add_argument('--no-sync', action='store_true', help='Salta l\'aggiornamento dei dati e del sentiment')
    args = parser.parse_args()

    if args.list:
        print("\n📋 Strategie disponibili:\n")
        for key, info in STRATEGIES.items():
            tipo = "📈 Azioni" if info['type'] == 'stock' else "🪙 Crypto"
            print(f"  {key:8s}  {tipo}  ({info['class']})")
        print(f"\nUso: python main.py --strategy {' '.join(STRATEGIES.keys())}")
        return

    # Determina quali strategie eseguire
    if args.all:
        keys = list(STRATEGIES.keys())
    elif args.stock:
        keys = [k for k, v in STRATEGIES.items() if v['type'] == 'stock']
    elif args.crypto:
        keys = [k for k, v in STRATEGIES.items() if v['type'] == 'crypto']
    else:
        keys = args.strategy

    # Aggiornamento preventivo di dati e notizie
    if not args.no_sync:
         logging.info("=== SINCRONIZZAZIONE MERCATO E NOTIZIE ===")
         try:
             # Sync mercato
             from sync_market_data import sync as sync_market
             sync_market(include_crypto=args.crypto or args.all or ('crypto' in keys))
         except Exception as e:
             logging.error(f"Errore durante l'aggiornamento dati/news: {e}")

    logging.info(f"=== AVVIO ROUTINE: {', '.join(keys)} ===")

    for key in keys:
        try:
            logging.info(f"⏳ Avvio {key}...")
            run_strategy(key)
            logging.info(f"✅ {key} completata.")
        except Exception as e:
            logging.error(f"❌ ERRORE in {key}: {e}", exc_info=True)
        finally:
            gc.collect()
            time.sleep(5)

    logging.info("=== ROUTINE TERMINATA ===")


if __name__ == "__main__":
    main()
