import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Crea cartelle se non esistono
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── DATABASE ──────────────────────────────────────────────
# Dati storici di mercato (condiviso tra tutte le strategie)
DB_MARKET = os.path.join(DATA_DIR, "market_data.db")
DB_MARKET_V70 = os.path.join(DATA_DIR, "market_data_v7_0.db")

# Stato operativo per ogni strategia (portafoglio, storico trade)
DB_TRADES_V43 = os.path.join(DATA_DIR, "trades_v4_3.db")
DB_TRADES_V46 = os.path.join(DATA_DIR, "trades_v4_6.db")
DB_TRADES_V56 = os.path.join(DATA_DIR, "trades_v5_6.db")
DB_TRADES_V64 = os.path.join(DATA_DIR, "trades_v6_4.db")
DB_TRADES_V70 = os.path.join(DATA_DIR, "trades_v7_0.db")

# Vecchi database (mantenuti per retrocompatibilità / migrazione)
DB_STOCK_V45 = os.path.join(DATA_DIR, "stock_data_v45.db")
DB_STOCK_V4 = os.path.join(DATA_DIR, "stock_data.db")

# ── TELEGRAM ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ── NEWS API ──────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '') # API opzionale per news
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')         # API principale per estrazione news (fallback)

# ── ALPACA — API key per strategia ────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")           # V4.3
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")     # V4.3

ALPACA_API_KEY_4_6 = os.getenv("ALPACA_API_KEY_4_6")
ALPACA_SECRET_KEY_4_6 = os.getenv("ALPACA_SECRET_KEY_4_6")

ALPACA_API_KEY_5_6 = os.getenv("ALPACA_API_KEY_5_6")
ALPACA_SECRET_KEY_5_6 = os.getenv("ALPACA_SECRET_KEY_5_6")

ALPACA_API_KEY_6_4 = os.getenv("ALPACA_API_KEY_6_4")
ALPACA_SECRET_KEY_6_4 = os.getenv("ALPACA_SECRET_KEY_6_4")

ALPACA_API_KEY_7 = os.getenv("ALPACA_API_KEY_7")             # V7.0 Intraday
ALPACA_SECRET_KEY_7 = os.getenv("ALPACA_SECRET_KEY_7")

# ── TICKERS ───────────────────────────────────────────────
TARGET_TICKERS_AZIONARIO = ['NVDA', 'META', 'AMZN', 'GOOGL', 'AAPL', 'MSFT', 'INTC', 'BA', 'PFE', 'PYPL', 'SBUX', 'NKE', 'TSLA', 'NFLX', 'DIS', 'JPM', 'V', 'XOM', 'WMT', 'KO']
TARGET_TICKERS_V43 = ['NVDA', 'META', 'NFLX', 'PYPL', 'BA', 'PFE']
BASE_TICKERS_V43 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'CVX', 'XOM', 'PFE', 'KO', 'PEP', 'CSCO']
TARGET_TICKERS_CRIPTO = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD']
TARGET_TICKERS_V70 = ['NVDA', 'META', 'AMZN', 'GOOGL', 'AAPL', 'MSFT', 'TSLA', 'NFLX', 'AMD', 'JPM']  # Alta liquidità per intraday

# ── MACRO MAPPING ─────────────────────────────────────────
MACRO_MAP = {'QQQ': 'nasdaq_close', '^VIX': 'vix_close', '^TNX': 'tnx_close', 'SOXX': 'soxx_close', 'GLD': 'gld_close'}
MACRO_LABELS_ORDERED = ['nasdaq_close', 'vix_close', 'tnx_close', 'soxx_close', 'gld_close']
