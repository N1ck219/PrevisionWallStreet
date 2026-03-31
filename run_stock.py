import time
import logging
import os
import sys
import gc

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - AZIONI - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "reports", "run_stocks_log.txt")),
        logging.StreamHandler()
    ]
)

# Importa le strategie azionarie
from strategies import strategy_v4_3
from strategies import strategy_v4_6
from strategies import strategy_v5_6
from strategies import strategy_v6_4

def main():
    logging.info("=== AVVIO ROUTINE AZIONARIO SEQUENZIALE ===")
    
    strategies = [
        ("V4.3", strategy_v4_3.run),
        ("V4.6", strategy_v4_6.run),
        ("V5.6", strategy_v5_6.run),
        ("V6.4", strategy_v6_4.run)
    ]
    
    for nome, func in strategies:
        try:
            logging.info(f"⏳ Avvio {nome}...")
            func()
            logging.info(f"✅ {nome} completata.")
        except Exception as e:
            logging.error(f"❌ ERRORE in {nome}: {e}", exc_info=True)
        finally:
            # Forza la pulizia della RAM/GPU prima di passare al modello successivo
            gc.collect()
            time.sleep(5) 
            
    logging.info("=== ROUTINE AZIONARIO TERMINATA ===")

if __name__ == "__main__":
    main()