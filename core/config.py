import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Crea cartelle se non esistono
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database
DB_STOCK_V45 = os.path.join(DATA_DIR, "stock_data_v45.db")
DB_STOCK_V4 = os.path.join(DATA_DIR, "stock_data.db")

# Telegram & API
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Tickers
TARGET_TICKERS_AZIONARIO = ['NVDA', 'META', 'AMZN', 'GOOGL', 'AAPL', 'MSFT', 'INTC', 'BA', 'PFE', 'PYPL', 'SBUX', 'NKE', 'TSLA', 'NFLX', 'DIS', 'JPM', 'V', 'XOM', 'WMT', 'KO']
TARGET_TICKERS_V43 = ['NVDA', 'META', 'NFLX', 'PYPL', 'BA', 'PFE']
BASE_TICKERS_V43 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'CVX', 'XOM', 'PFE', 'KO', 'PEP', 'CSCO']
TARGET_TICKERS_CRIPTO = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD']

# Macro mapping
MACRO_MAP = {'QQQ': 'nasdaq_close', '^VIX': 'vix_close', '^TNX': 'tnx_close', 'SOXX': 'soxx_close', 'GLD': 'gld_close'}
MACRO_LABELS_ORDERED = ['nasdaq_close', 'vix_close', 'tnx_close', 'soxx_close', 'gld_close']
