import requests
import datetime
from core.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TelegramNotifier:
    @staticmethod
    def send_message(testo_html):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
            return
        # Rimuovi doppi a capo extra per pulizia
        testo_html = testo_html.replace(r'\n', '\n')
        try: 
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": testo_html, "parse_mode": "HTML"}, 
                          timeout=10)
        except Exception as e: 
            print(f"Errore invio telegram: {e}")

    @staticmethod
    def build_report(bot_name, balance_str="", win_rate_str="", trades_str="", extra_str="", logs_str=""):
        oggi = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        msg = f"🏦 <b>{bot_name}</b>\n📅 Data: {oggi}\n──────────────\n"
        
        has_stats = False
        if balance_str:
            msg += f"💼 <b>BILANCIO:</b> {balance_str}\n"
            has_stats = True
        if win_rate_str:
            msg += f"📊 <b>WIN RATE:</b> {win_rate_str}\n"
            has_stats = True
        if extra_str:
            msg += f"ℹ️ {extra_str}\n"
            has_stats = True
            
        if has_stats:
            msg += "──────────────\n"
            
        msg += f"🔔 <b>SEGNALI / ORDINI:</b>\n"
        if trades_str:
            msg += f"{trades_str}\n"
        else:
            msg += "⚠️ Nessuna nuova operazione oggi.\n"
            
        if logs_str:
            msg += "──────────────\n"
            msg += f"📋 <b>LOGS / CHIUSURE:</b>\n{logs_str}\n"
            
        return msg
