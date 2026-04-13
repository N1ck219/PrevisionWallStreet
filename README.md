# 📈 PrevisionWallStreet

Bot di trading algoritmico multi-strategia basato su reti neurali **LSTM con meccanismo di Attention**, progettato per generare segnali operativi su **azioni USA** e **criptovalute**.

Il sistema scarica dati di mercato via [yfinance](https://github.com/ranaroussi/yfinance), li arricchisce con indicatori tecnici e dati macroeconomici, e produce previsioni direzionali inviate in tempo reale tramite **Telegram**.

---

## ✨ Funzionalità principali

| Feature | Dettaglio |
|---|---|
| **Multi-strategia** | 4 strategie azionarie (V4.3 → V6.4) + 1 crypto (V1.7) |
| **Modelli LSTM + Attention** | Architetture single-input e *Split Brain* (doppio input) |
| **Dati macro** | Integrazione NASDAQ, VIX, TNX, SOXX, GLD |
| **Backtesting** | Simulazione storica per ogni strategia (`simulations/`) |
| **Database separati** | Dati di mercato condivisi + DB operativo per ogni strategia |
| **Sync S&P 500** | Script dedicato per scaricare lo storico completo dal 1990 |
| **Notifiche Telegram** | Report unificati HTML con segnali, bilancio e statistiche |
| **Esecuzione ordini** | Integrazione opzionale con **Alpaca Markets** e **Binance** |

---

## 🏗️ Architettura del progetto

```
PrevisionWallStreet/
├── core/                       # Moduli condivisi
│   ├── base_strategy.py        # Classe base astratta per tutte le strategie
│   ├── config.py               # Configurazione globale (tickers, percorsi, API keys, DB)
│   ├── data_manager.py         # Download e caching dati mercato (yfinance → SQLite)
│   ├── features.py             # Feature engineering: indicatori tecnici + macro + crypto
│   ├── model_factory.py        # Factory pattern per istanziare i modelli LSTM
│   └── notifier.py             # Notifiche Telegram (report builder unificato)
│
├── strategies/                 # Strategie di trading live (estendono BaseStrategy)
│   ├── strategy_v4_3.py        # StrategyV43 — LSTM base con Attention
│   ├── strategy_v4_6.py        # StrategyV46 — LSTM profonda (128 neuroni)
│   ├── strategy_v5_6.py        # StrategyV56 — Split Brain (tecnico + macro)
│   ├── strategy_v6_4.py        # StrategyV64 — Apex Fund, gestione rischio avanzata
│   └── strategy_crypto_v1_7.py # StrategyCryptoV17 — Criptovalute
│
├── simulations/                # Backtesting per ogni strategia
│   ├── backtest_v4_3.py
│   ├── backtest_v4_6.py
│   ├── backtest_v5_6.py
│   ├── backtest_v6_4.py
│   └── backtest_crypto_v1_7.py
│
├── models/                     # Pesi dei modelli addestrati (.h5)
│
├── data/                       # Database SQLite
│   ├── market_data.db          # Dati storici OHLCV (condiviso, S&P 500 dal 1990)
│   ├── trades_v4_3.db          # Stato portafoglio e storico operazioni V4.3
│   ├── trades_v4_6.db          # Stato portafoglio e storico operazioni V4.6
│   ├── trades_v5_6.db          # Stato portafoglio V5.6
│   └── trades_v6_4.db          # Stato portafoglio V6.4
│
├── main.py                     # CLI unificato (--all, --stock, --crypto, --strategy, --no-sync)
├── run_stock.py                # Entry point legacy — strategie azionarie
├── run_crypto.py               # Entry point legacy — strategia crypto
├── sync_market_data.py         # Aggiornamento dati di mercato nel DB condiviso
├── migrate_databases.py        # Migrazione una tantum dai vecchi DB ai nuovi
└── requirements.txt            # Dipendenze Python
```

---

## 🗄️ Architettura Database

I database sono separati per responsabilità:

| Database | Contenuto | Usato da |
|---|---|---|
| `market_data.db` | Dati storici OHLCV (cache yfinance) | Tutte le strategie (lettura) |
| `trades_v4_3.db` | Portafoglio, storico operazioni, benchmark | Solo V4.3 |
| `trades_v4_6.db` | Posizioni live, storico win/loss | Solo V4.6 |
| `trades_v5_6.db` | Stato portafoglio (entry, ATR, qty) | Solo V5.6 |
| `trades_v6_4.db` | Stato portafoglio (entry, invested, qty) | Solo V6.4 |

**Vantaggi:**
- I dati di mercato vengono scaricati una sola volta e condivisi
- Ogni strategia può essere resettata indipendentemente
- Nessun conflitto di scrittura tra strategie diverse

---

## 🧠 Architetture dei modelli

### Single-Input (V4.3, V4.6, Crypto V1.7)

Sequenza temporale di feature tecniche → **LSTM → Attention → LSTM → Dense → output sigmoide**.

### Split Brain (V5.6, V6.4)

 Due rami paralleli:
- **Ramo tecnico** — Feature del singolo ticker (RSI, Bollinger, ATR, OBV, ecc.)
- **Ramo macro** — Rendimenti degli indici macroeconomici (NASDAQ, VIX, TNX, SOXX, GLD)

I due rami vengono concatenati prima del layer Dense finale.


---

## ⚙️ Setup

### 1. Clona il repository

```bash
git clone https://github.com/N1ck219/PrevisionWallStreet.git
cd PrevisionWallStreet
```

### 2. Crea il virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura le variabili d'ambiente

Crea un file `.env` nella root del progetto:

```env
TELEGRAM_BOT_TOKEN=il_tuo_token
TELEGRAM_CHAT_ID=il_tuo_chat_id

# Alpaca — una coppia di chiavi per ogni strategia (paper trading)
ALPACA_API_KEY=...           # V4.3
ALPACA_SECRET_KEY=...
ALPACA_API_KEY_4_6=...
ALPACA_SECRET_KEY_4_6=...
ALPACA_API_KEY_5_6=...
ALPACA_SECRET_KEY_5_6=...
ALPACA_API_KEY_6_4=...
ALPACA_SECRET_KEY_6_4=...

# Binance — per la strategia crypto (opzionale)
BINANCE_API_KEY=...
BINANCE_SECRET=...
```

---

## 🚀 Utilizzo

### 1. Sincronizza i dati di mercato

```bash
# Aggiornamento rapido (solo ticker del progetto, dal 2020)
python sync_market_data.py

# Download completo S&P 500 dal 1990 (per addestramento modelli)
python sync_market_data.py --full

# Includi anche le criptovalute
python sync_market_data.py --crypto

# Tutto insieme con output dettagliato
python sync_market_data.py --full --crypto -v
```

### 2. Esecuzione strategie
Lanciando il runner integrato, il programma per prima cosa esegue il sync automtico di mercato e notizie, poi avvia le versioni del bot scelte in riga di comando.

```bash
# Tutte le strategie (azioni + crypto)
python main.py --all

# Salta la lunga procedura di sincronizzazione e vai dritto alle strategie
python main.py --all --no-sync

# Solo azioni (V4.3 → V6.4)
python main.py --stock

# Solo crypto
python main.py --crypto

# Strategie specifiche
python main.py --strategy v4.3 v6.4

# Lista strategie disponibili
python main.py --list
```

Compatibile con **crontab**:
```bash
# Ogni giorno alle 22:00 (lunedì-venerdì)
0 22 * * 1-5 cd /path/to/PrevisionWallStreet && python main.py --stock
# Ogni 4 ore per crypto
0 */4 * * * cd /path/to/PrevisionWallStreet && python main.py --crypto
```

> **Nota:** `run_stock.py` e `run_crypto.py` sono ancora funzionanti per retrocompatibilità.

### 3. Backtesting

```bash
python simulations/backtest_v6_4.py
```

### 4. Migrazione (una tantum)

Se hai dati nei vecchi database (`stock_data.db`, `stock_data_v45.db`):

```bash
python migrate_databases.py
```

---

## 📊 Indicatori tecnici utilizzati

| Indicatore | Descrizione |
|---|---|
| **RSI 14** | Relative Strength Index a 14 periodi |
| **Bollinger %B** | Posizione del prezzo rispetto alle bande di Bollinger |
| **Bollinger Width** | Ampiezza delle bande di Bollinger |
| **ATR 14** | Average True Range a 14 periodi |
| **SMA 50 / 200** | Medie mobili semplici e distanza percentuale |
| **OBV** | On-Balance Volume (variazione percentuale) |

---

## 🔧 Stack tecnologico

- **Python 3.10+**
- **TensorFlow / Keras** — Modelli LSTM + Attention
- **yfinance** — Download dati di mercato
- **SQLite** — Database locale (dati di mercato + operazioni)
- **Alpaca Markets SDK** — Esecuzione ordini azioni (paper trading)
- **CCXT / Binance** — Esecuzione ordini crypto (futures)
- **scikit-learn** — Preprocessing (StandardScaler)
- **Telegram Bot API** — Notifiche in tempo reale

---

## ⚠️ Disclaimer

> Questo progetto è a scopo **educativo e di ricerca**. Non costituisce consulenza finanziaria.
> Il trading comporta rischi significativi, inclusa la possibile perdita del capitale investito.
> L'autore non è responsabile per eventuali perdite derivanti dall'uso di questo software.

---

## 📄 Licenza

Distribuito con licenza **MIT**. Vedi il file [LICENSE](LICENSE) per maggiori dettagli.
