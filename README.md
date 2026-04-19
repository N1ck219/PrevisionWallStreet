# 📈 PrevisionWallStreet — Advanced Trading System

Un sistema di trading algoritmico modulare e multi-strategia che utilizza architetture avanzate di Deep Learning (**LSTM**, **Multi-Head Attention** ed **Informer**) per prevedere l'andamento di Azioni USA e Criptovalute.

Il bot integra indicatori tecnici, trend macroeconomici e analisi statistica per generare segnali direzionali, gestendo l'intero ciclo di vita: dalla raccolta dati (yfinance/Alpaca) all'addestramento, fino all'esecuzione e notifica su Telegram.

---

## 🚀 Funzionalità Chiave

*   **Modularità Totale**: Architettura pulita con separazione netta tra logica Core, Strategie, Simulazioni e Script operativi.
*   **Deep Learning Multi-Modello**:
    *   **V4/V6**: LSTM + Self-Attention per trend daily.
    *   **V7 (Split Brain)**: Modello a due rami (Tecnico + Macro) ottimizzato per l'intraday a 5 minuti.
    *   **V8 (Informer)**: Architettura allo stato dell'arte per serie temporali, progettata per catturare dipendenze a lungo termine con efficienza computazionale.
*   **Gestione Dati Efficiente**: Utilizzo di database SQLite centralizzati per il mercato e database dedicati per ogni strategia (Trade DBs) per evitare conflitti.
*   **Dataset HDF5**: Supporto per il caricamento di grandi volumi di dati intraday tramite generatori HDF5 per ottimizzare l'uso della RAM durante il training.
*   **Notifiche Real-time**: Report dettagliati dei segnali e performance inviati via Telegram.

---

## 🏗️ Architettura del Progetto

```text
PrevisionWallStreet/
├── core/                       # Motore applicativo (Core Engine)
│   ├── models/                 # Definizioni delle architetture neurali (Informer, LSTM, ecc.)
│   ├── data/                   # Logica di processing: feature engineering e data management
│   ├── utils/                  # Servizi ausiliari: notifiche Telegram e analisi sentiment
│   ├── database/               # Utility per migrazioni e manutenzione DB
│   └── config.py               # Configurazione centralizzata (API, Path, Tickers)
├── strategies/                 # Implementazioni delle strategie LIVE
│   ├── v4/ v5/ v6/ v7/ v8/     # Strategie raggruppate per versione
│   ├── crypto/                 # Strategia specializzata per Criptovalute
│   └── base_strategy.py        # Classe base astratta per tutte le strategie
├── simulations/                # Ambienti di Backtesting
│   ├── v7/ v8/ ...             # Backtest speculari alle versioni live
├── scripts/                    # Strumenti operativi e di manutenzione
│   ├── training/               # Script per l'addestramento dei modelli (v7, v8)
│   ├── maintenance/            # Aggiornamento dati (sync) e download intraday
│   └── runners/                # Entry-points manuali per singoli moduli
├── data/                       # Archiviazione fisica
│   ├── databases/              # SQLite DB (Market Data e Trade Logs)
│   └── datasets/               # File HDF5 e Scaler (.pkl) per il training
├── models/                     # Pesi dei modelli salvati (.h5, .keras)
├── reports/                    # Log di esecuzione e report grafici suddivisi per versione
├── main.py                     # Entry-point UNIFICATO per il trading live
└── requirements.txt            # Dipendenze del progetto
```

---

## 🛠️ Setup e Installazione

1.  **Requisiti**: Python 3.10+ e una GPU NVIDIA (consigliata per V7/V8).
2.  **Installazione**:
    ```bash
    git clone https://github.com/N1ck219/PrevisionWallStreet.git
    cd PrevisionWallStreet
    python -m venv .venv
    source .venv/bin/activate  # .venv\Scripts\activate su Windows
    pip install -r requirements.txt
    ```
3.  **Configurazione**: Crea un file `.env` nella root:
    ```env
    TELEGRAM_BOT_TOKEN=tuo_token
    TELEGRAM_CHAT_ID=tua_chat_id
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    # Chiavi specifiche per versioni (es. ALPACA_API_KEY_7)
    ```

---

## 📈 Flusso di Lavoro Operativo

### 1. Gestione Dati
Per mantenere il sistema aggiornato, utilizza gli script di manutenzione:
```bash
# Sincronizza i dati giornalieri e macro
python scripts/maintenance/sync_market_data.py --crypto

# Scarica dati intraday (1-min) per le versioni avanzate
python scripts/maintenance/download_intraday_data.py
```

### 2. Addestramento Modelli
Gli script di training sono isolati per evitare confusione:
```bash
# Prepara il dataset HDF5 per la V8
python scripts/training/prepare_v8_data.py

# Avvia l'addestramento
python scripts/training/train_v8_0.py --epochs 50 --batch-size 256
```

### 3. Trading Live
Usa l'entry-point unificato per gestire tutte le strategie:
```bash
# Mostra le strategie disponibili
python main.py --list

# Esegui una strategia specifica
python main.py --strategy v7.3

# Esegui tutte le strategie (Live Mode)
python main.py --all
```

---

## 📊 Backtesting
Ogni strategia può essere validata storicamente prima del deploy:
```bash
python simulations/v7/backtest_v7_3.py --days 100
```

---

## ⚠️ Disclaimer
Questo software è a scopo puramente **educativo e di ricerca**. Il trading comporta rischi reali di perdita di capitale. L'autore non si assume alcuna responsabilità per l'uso improprio o per perdite finanziarie derivanti dall'utilizzo di questi algoritmi.
