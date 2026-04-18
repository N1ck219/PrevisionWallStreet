import time
import logging
import os
import sys
import gc
import datetime

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - LIVE V7.2 - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "reports", "live_v7_2_log.txt"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Forza l'output del terminale in UTF-8 se possibile (utile su Windows)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

from strategies.v7.strategy_v7_2 import StrategyV72

def main():
    logging.info("=== AVVIO BOT V7.2 IN LIVE LOOP ===")
    logging.info("Il bot opererà automaticamente ogni minuto durante le ore di mercato (9:30 - 16:00 ET).")
    logging.info("Per terminarlo, premi CTRL+C in questo terminale.")
    
    bot = StrategyV72()
    
    while True:
        try:
            now_et = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))
            weekday = now_et.weekday()
            
            # Non gira nel weekend
            if weekday >= 5:
                logging.info(f"Weekend. Il bot riposa. Prossimo check tra 1 ora.")
                time.sleep(3600)
                continue
            
            # Orario di mercato: 9:30 ET (570) - 16:00 ET (960)
            total_minutes = now_et.hour * 60 + now_et.minute
            
            if 570 <= total_minutes <= 960:
                logging.info(f"⏳ Avvio esecuzione V7.2...")
                
                # Esegue la strategia aggiornando DB e log
                bot.run()
                
                logging.info(f"✅ Ciclo V7.2 completato.")
                
            else:
                logging.info(f"Mercato chiuso (ET: {now_et.strftime('%H:%M')}). In attesa...")
                # Se è notte o fuori orario, pausa più lunga (es. 5 minuti)
                time.sleep(300)
                continue
            
            # Pulizia memoria profonda
            gc.collect()
            
            # Allineamento con il clock: dorme fino all'inizio del minuto successivo
            ora = datetime.datetime.now()
            # Calcola quanti secondi mancano allo scoccare del minuto, più 1 per sicurezza
            sleep_time = 60 - ora.second + 2 
            
            logging.info(f"Attesa di {sleep_time}s per il prossimo controllo al minuto esatto.")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logging.info("=== ARRESTO MANUALE RICEVUTO (CTRL+C) ===")
            break
        except Exception as e:
            logging.error(f"❌ ERRORE nel loop: {e}", exc_info=True)
            # In caso di errore anomalo, ferma 30 sec e riprova
            time.sleep(30)

if __name__ == "__main__":
    main()
