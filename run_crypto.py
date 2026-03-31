import logging
import os
import sys

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - CRIPTO - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "reports", "run_crypto_log.txt")),
        logging.StreamHandler()
    ]
)

# Importa la strategia cripto
from strategies import strategy_crypto_v1_7

def main():
    logging.info("=== AVVIO ROUTINE CRIPTO ===")
    try:
        logging.info("⏳ Avvio Crypto V1.7...")
        strategy_crypto_v1_7.run()
        logging.info("✅ Crypto V1.7 completata.")
    except Exception as e:
        logging.error(f"❌ ERRORE in Crypto V1.7: {e}", exc_info=True)
    logging.info("=== ROUTINE CRIPTO TERMINATA ===")

if __name__ == "__main__":
    main()