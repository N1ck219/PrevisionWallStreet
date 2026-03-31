# 📈 PrevisionWallStreet

Bot di trading algoritmico multi-strategia basato su reti neurali **LSTM con meccanismo di Attention**, progettato per generare segnali operativi su **azioni USA** e **criptovalute**.

Il sistema scarica dati di mercato via [yfinance](https://github.com/ranaroussi/yfinance), li arricchisce con indicatori tecnici e dati macroeconomici, e produce previsioni direzionali inviate in tempo reale tramite **Telegram**.

---

## ✨ Funzionalità principali

| Feature | Dettaglio |
|---|---|
| **Multi-strategia** | 4 strategie azionarie (V4.3 → V6.4) + 1 crypto (V1.7) |
| **Modelli LSTM + Attention** | Architetture single-input e *Split Brain* (doppio input tecnico/macro) |
| **Dati macro integrati** | NASDAQ, VIX, TNX, SOXX, GLD come feature aggiuntive |
| **Backtesting** | Simulazione storica per ogni strategia (`simulations/`) |
| **Cache SQLite** | Dati storici salvati in locale per minimizzare le chiamate API |
| **Notifiche Telegram** | Report formattati HTML con segnali, bilancio e statistiche |
| **Esecuzione ordini** | Integrazione opzionale con **Alpaca Markets** (paper trading) |

---

## 🏗️ Architettura del progetto

```
PrevisionWallStreet/
├── core/                       # Moduli condivisi
│   ├── config.py               # Configurazione globale (tickers, percorsi, API keys)
│   ├── data_manager.py         # Download e caching dati mercato (yfinance → SQLite)
│   ├── features.py             # Feature engineering: indicatori tecnici + macro
│   ├── model_factory.py        # Factory pattern per istanziare i modelli LSTM
│   └── notifier.py             # Notifiche Telegram (report builder)
│
├── strategies/                 # Strategie di trading live
│   ├── strategy_v4_3.py        # LSTM base con Attention
│   ├── strategy_v4_6.py        # LSTM profonda (128 neuroni)
│   ├── strategy_v5_6.py        # Split Brain (tecnico + macro)
│   ├── strategy_v6_4.py        # Apex Fund — gestione rischio avanzata
│   └── strategy_crypto_v1_7.py # Strategia dedicata criptovalute
│
├── simulations/                # Backtesting per ogni strategia
│   ├── backtest_v4_3.py
│   ├── backtest_v4_6.py
│   ├── backtest_v5_6.py
│   ├── backtest_v6_4.py
│   └── backtest_crypto_v1_7.py
│
├── models/                     # Pesi dei modelli addestrati (.h5)
├── data/                       # Database SQLite con dati storici
├── run_stock.py                # Entry point — esecuzione strategie azionarie
├── run_crypto.py               # Entry point — esecuzione strategia crypto
└── requirements.txt            # Dipendenze Python
```

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
git clone https://github.com/<tuo-username>/PrevisionWallStreet.git
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

# Opzionale — per esecuzione ordini su Alpaca (paper trading)
ALPACA_API_KEY=la_tua_api_key
ALPACA_SECRET_KEY=la_tua_secret_key
```

---

## 🚀 Utilizzo

### Esecuzione strategie azionarie

```bash
python run_stock.py
```

Esegue in sequenza: V4.3 → V4.6 → V5.6 → V6.4, con pulizia della memoria tra un modello e l'altro.

### Esecuzione strategia crypto

```bash
python run_crypto.py
```

### Backtesting

```bash
python simulations/backtest_v6_4.py
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
- **SQLite** — Cache dati storici locale
- **Alpaca Markets SDK** — Esecuzione ordini (paper trading)
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
