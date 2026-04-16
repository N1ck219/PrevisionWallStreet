# 📈 PrevisionWallStreet

Bot di trading algoritmico multi-strategia basato su reti neurali **LSTM con meccanismo di Attention**, progettato per generare segnali operativi su **azioni USA** e **criptovalute**.

Il sistema scarica dati di mercato via [yfinance](https://github.com/ranaroussi/yfinance) e [Alpaca Markets](https://alpaca.markets/), li arricchisce con indicatori tecnici e dati macroeconomici, e produce previsioni direzionali inviate in tempo reale tramite **Telegram**.

---

## ✨ Funzionalità principali

| Feature | Dettaglio |
|---|---|
| **Multi-strategia** | 5 strategie azionarie (V4.3 → V7.0) + 1 crypto (V1.7) |
| **Modelli LSTM + Attention** | Architetture single-input, *Split Brain* e *Split Brain V7.0* con Multi-Head Attention |
| **Intraday V7.0** | Analisi a 1 minuto, previsioni a 1 ora, VWAP e feature di sessione |
| **Dati macro** | Integrazione NASDAQ, VIX, TNX, SOXX, GLD |
| **Backtesting** | Simulazione storica per ogni strategia (`simulations/`) |
| **Database separati** | Dati di mercato condivisi + DB operativo per ogni strategia |
| **Sync S&P 500** | Script dedicato per scaricare lo storico completo dal 1990 |
| **Download Intraday** | Script per dati a 1 minuto da Alpaca Markets (~7 anni) |
| **Notifiche Telegram** | Report unificati HTML con segnali, bilancio e statistiche |
| **Esecuzione ordini** | Integrazione con **Alpaca Markets** e **Binance** |

---

## 🏗️ Architettura del progetto

```
PrevisionWallStreet/
├── core/                       # Moduli condivisi
│   ├── base_strategy.py        # Classe base astratta per tutte le strategie
│   ├── config.py               # Configurazione globale (tickers, percorsi, API keys, DB)
│   ├── data_manager.py         # Download e caching dati mercato (yfinance + Alpaca → SQLite)
│   ├── features.py             # Feature engineering: indicatori tecnici + VWAP + macro + session
│   ├── model_factory.py        # Factory pattern per istanziare i modelli LSTM
│   └── notifier.py             # Notifiche Telegram (report builder unificato)
│
├── strategies/                 # Strategie di trading live (estendono BaseStrategy)
│   ├── strategy_v4_3.py        # StrategyV43 — LSTM base con Attention
│   ├── strategy_v4_6.py        # StrategyV46 — LSTM profonda (128 neuroni)
│   ├── strategy_v5_6.py        # StrategyV56 — Split Brain (tecnico + macro)
│   ├── strategy_v6_4.py        # StrategyV64 — Apex Fund, gestione rischio avanzata
│   ├── strategy_v7_0.py        # StrategyV70 — Intraday Sniper, Split Brain V7.0 + VWAP
│   └── strategy_crypto_v1_7.py # StrategyCryptoV17 — Criptovalute
│
├── simulations/                # Backtesting per ogni strategia
│   ├── backtest_v4_3.py
│   ├── backtest_v4_6.py
│   ├── backtest_v5_6.py
│   ├── backtest_v6_4.py
│   ├── backtest_v7_0.py        # Backtest intraday (ultimi 500 giorni)
│   └── backtest_crypto_v1_7.py
│
├── models/                     # Pesi dei modelli addestrati (.h5)
│
├── data/                       # Database SQLite
│   ├── market_data.db          # Dati storici OHLCV daily + intraday 1-min (condiviso)
│   ├── trades_v4_3.db          # Stato portafoglio e storico operazioni V4.3
│   ├── trades_v4_6.db          # Stato portafoglio e storico operazioni V4.6
│   ├── trades_v5_6.db          # Stato portafoglio V5.6
│   ├── trades_v6_4.db          # Stato portafoglio V6.4
│   └── trades_v7_0.db          # Stato portafoglio + storico trade V7.0
│
├── main.py                     # CLI unificato (--all, --stock, --crypto, --strategy, --no-sync)
├── download_intraday_data.py   # Download massivo dati 1-min da Alpaca Markets
├── train_v7_0.py               # Training V7.0 (pre-train daily + fine-tune intraday)
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
| `market_data.db` | Dati storici OHLCV daily (cache yfinance) + intraday 1-min (Alpaca) | Tutte le strategie (lettura) |
| `trades_v4_3.db` | Portafoglio, storico operazioni, benchmark | Solo V4.3 |
| `trades_v4_6.db` | Posizioni live, storico win/loss | Solo V4.6 |
| `trades_v5_6.db` | Stato portafoglio (entry, ATR, qty) | Solo V5.6 |
| `trades_v6_4.db` | Stato portafoglio (entry, invested, qty) | Solo V6.4 |
| `trades_v7_0.db` | Posizioni intraday + storico trade con motivo chiusura | Solo V7.0 |

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

### Split Brain V7.0 (Intraday)

Evoluzione dell'architettura Split Brain ottimizzata per dati a 1 minuto con:
- **Ramo tecnico** — LSTM(256) → Multi-Head Attention (2 heads) → LayerNorm + Residual → LSTM(128)
  - Include: RSI, Bollinger, ATR, OBV, **VWAP ratio**, minuti dall'apertura, percentuale sessione
- **Ramo macro** — LSTM(64) con rendimenti macro + flag sessione (power hour, opening range)
- **Merger** — Dense(128) → BatchNorm → Dense(64) → output sigmoide

**Training a due fasi:**
1. **Pre-training** su dati daily (2000-2019) per catturare pattern macro a lungo termine
2. **Fine-tuning** su dati intraday 1-min (2019-oggi) per specializzazione intraday


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
ALPACA_API_KEY_7=...         # V7.0 Intraday
ALPACA_SECRET_KEY_7=...

# Binance — per la strategia crypto (opzionale)
BINANCE_API_KEY=...
BINANCE_SECRET=...
```

---

## 🚀 Utilizzo

### 1. Sincronizza i dati di mercato (daily)

```bash
# Aggiornamento rapido (solo ticker del progetto, dal 2020)
python sync_market_data.py

# Download completo S&P 500 dal 1990 (per addestramento modelli)
python sync_market_data.py --full

# Includi anche le criptovalute
python sync_market_data.py --crypto
```

### 2. Download dati intraday per V7.0

```bash
# Download dati a 1 minuto da Alpaca (~7 anni, tutti i ticker V7.0)
python download_intraday_data.py

# Stato del download
python download_intraday_data.py --status

# Download singolo ticker
python download_intraday_data.py --ticker AAPL

# Includi anche i ticker macro (QQQ, GLD, SOXX)
python download_intraday_data.py --include-macro
```

### 3. Training modello V7.0

```bash
# Pipeline completa (pre-train daily + fine-tune intraday)
python train_v7_0.py

# Solo pre-training su dati daily
python train_v7_0.py --phase pretrain

# Solo fine-tuning su dati intraday
python train_v7_0.py --phase finetune

# Personalizza epoche
python train_v7_0.py --epochs-pretrain 100 --epochs-finetune 50
```

### 4. Esecuzione strategie

```bash
# Tutte le strategie (azioni + crypto)
python main.py --all

# Solo la V7.0 intraday
python main.py --strategy v7.0

# Solo azioni daily (V4.3 → V6.4)
python main.py --stock

# Lista strategie disponibili
python main.py --list

# Salta sincronizzazione
python main.py --all --no-sync
```

Compatibile con **crontab**:
```bash
# Ogni giorno alle 22:00 (lunedì-venerdì) — strategie daily
0 22 * * 1-5 cd /path/to/PrevisionWallStreet && python main.py --stock
# Ogni ora durante orari di mercato — V7.0 intraday
30 10-15 * * 1-5 cd /path/to/PrevisionWallStreet && python main.py --strategy v7.0 --no-sync
# Ogni 4 ore per crypto
0 */4 * * * cd /path/to/PrevisionWallStreet && python main.py --crypto
```

> **Nota:** `run_stock.py` e `run_crypto.py` sono ancora funzionanti per retrocompatibilità.

### 5. Backtesting

```bash
# Backtest intraday V7.0 (ultimi 500 giorni)
python simulations/backtest_v7_0.py

# Numero di giorni personalizzato
python simulations/backtest_v7_0.py --days 250

# Backtest altre strategie
python simulations/backtest_v6_4.py
```

### 6. Migrazione (una tantum)

Se hai dati nei vecchi database (`stock_data.db`, `stock_data_v45.db`):

```bash
python migrate_databases.py
```

---

## 📊 Indicatori tecnici utilizzati

| Indicatore | Descrizione | Versioni |
|---|---|---|
| **RSI 14** | Relative Strength Index a 14 periodi | Tutte |
| **Bollinger %B** | Posizione del prezzo rispetto alle bande di Bollinger | Tutte |
| **Bollinger Width** | Ampiezza delle bande di Bollinger | Tutte |
| **ATR 14** | Average True Range a 14 periodi | Tutte |
| **SMA 50 / 200** | Medie mobili semplici e distanza percentuale | Tutte |
| **OBV** | On-Balance Volume (variazione percentuale) | Tutte |
| **VWAP** | Volume Weighted Average Price con reset di sessione | V7.0 |
| **VWAP Ratio** | Posizione relativa del prezzo rispetto al VWAP | V7.0 |
| **Session Features** | Minuti dall'apertura, % sessione, power hour, opening range | V7.0 |

---

## 🔧 Stack tecnologico

- **Python 3.10+**
- **TensorFlow / Keras** — Modelli LSTM + Attention + Multi-Head Attention
- **yfinance** — Download dati di mercato daily
- **Alpaca Markets SDK** — Download dati intraday 1-min + esecuzione ordini (paper trading)
- **SQLite** — Database locale (dati di mercato + operazioni)
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
