import os
import sys
import sqlite3
import argparse
import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.config import DB_MARKET_V7, TARGET_TICKERS_AZIONARIO
from core.data_manager import DataManager
from core.news_fetcher import NewsFetcher
from core.sentiment_analyzer import SentimentEngine

def create_sentiment_table(conn):
    """Crea la tabella sentiment_cache se non esiste."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            Date TEXT, 
            Ticker TEXT, 
            Sentiment_Score REAL, 
            Confidence REAL, 
            Volatility REAL, 
            News_Count INTEGER,
            PRIMARY KEY (Date, Ticker)
        )
    ''')
    conn.commit()

def sync_daily_sentiment(target_tickers, days_back=7):
    """
    Sincronizza il sentiment quotidiano tramite YFinance/Finnhub/AlphaVantage.
    """
    print(f"📡 SYNC SENTIMENT QUOTIDIANO — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    conn = DataManager.setup_db(DB_MARKET_V7)
    create_sentiment_table(conn)
    
    # Inizializza il modello FinBERT
    engine = SentimentEngine()
    
    # Raggruppa le news raccolte per data e per ticker {ticker: {date: [news1, news2]}}
    news_batch_aggregate = {}
    
    for ticker in tqdm(target_tickers, desc="Recupero News Recenti"):
        news_items = NewsFetcher.get_recent_news(ticker, force_all=True, days_back=days_back)
        news_batch_aggregate[ticker] = {}
        for item in news_items:
             d = item['date']
             if d not in news_batch_aggregate[ticker]:
                  news_batch_aggregate[ticker][d] = []
             news_batch_aggregate[ticker][d].append(item)
    
    # Analisi del sentiment
    successi = 0
    nuovi_dati = 0
    
    for ticker, ticker_data in tqdm(news_batch_aggregate.items(), desc="Analisi con FinBERT"):
        print(f"\n📊 [{ticker}] Analisi completata per {len(ticker_data)} giorni.")
        for d, items in ticker_data.items():
            # Controlla se abbiamo già i dati di sentiment calcolati nel db (aggiorniamo solo se mancano)
            exists = conn.execute("SELECT News_Count FROM sentiment_cache WHERE Ticker=? AND Date=?", (ticker, d)).fetchone()
            
            # Se esiste e le news count analizzate è >= a quelle nuove trovate, saltiamo l'analisi.
            if exists and exists[0] >= len(items):
                continue
                
            score, confidence, volatility = engine.compute_daily_aggregate(items)
            
            # Stampa il resoconto dettagliato se è un sync con poche news (aiuta l'utente a capire cosa succede)
            if len(items) > 0 and len(ticker_data) < 100:
                print(f"   📅 {d} -> Score FinBERT: {score:+.3f} (Confidenza: {confidence:.2f}, Volatilità: {volatility:.2f})")
                print(f"      📌 Trovate {len(items)} notizie:")
                for idx, itm in enumerate(items[:3]): # Stampa al massimo le prime 3 per non inondare il log
                    print(f"      - {itm['title']}")
                if len(items) > 3:
                    print(f"      - ... e altre {len(items)-3} notizie.")
                    
            try:
                # Salvataggio nel DB
                conn.execute('''
                    INSERT OR REPLACE INTO sentiment_cache (Date, Ticker, Sentiment_Score, Confidence, Volatility, News_Count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (d, ticker, float(score), float(confidence), float(volatility), len(items)))
                nuovi_dati += 1
            except Exception as e:
                print(f"Errore salvataggio {ticker}: {e}")
        successi += 1
        
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"✅ SYNC SENTIMENT COMPLETATA")
    print(f"   Ticker elaborati: {successi}/{len(target_tickers)}")
    print(f"   Incroci Data/Ticker inseriti o aggiornati: {nuovi_dati}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days-back', type=int, default=7, help="Giorni di news da recuperare tramite API (default 7)")
    args = parser.parse_args()
    
    sync_daily_sentiment(TARGET_TICKERS_AZIONARIO, days_back=args.days_back)
