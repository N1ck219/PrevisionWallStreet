import os
import sys
import argparse
from tqdm import tqdm
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.config import DB_MARKET_V7, TARGET_TICKERS_AZIONARIO
from core.data_manager import DataManager
from core.news_fetcher import NewsFetcher
from core.sentiment_analyzer import SentimentEngine
import sqlite3

def backfill_from_kaggle(csv_path, target_tickers=TARGET_TICKERS_AZIONARIO, chunk_size=50000):
    """
    Legge un CSV Kaggle di news a blocchi, processa i titoli con FinBERT 
    e li salva nel DB sentiment_cache. Supporta il ticker GLOBAL.
    """
    print(f"\n" + "=" * 60)
    print(f"🔄 AVVIO IMPORTAZIONE MASSIVA: {os.path.basename(csv_path)}")
    print(f"🎯 Target Tickers: {len(target_tickers) if target_tickers else 'TUTTI'}")
    print(f"📦 Modalità: Memory-Safe (Chunks di {chunk_size} righe)")
    print("=" * 60 + "\n")
    
    conn = DataManager.setup_db(DB_MARKET_V7)
    from sync_v7_sentiment import create_sentiment_table
    create_sentiment_table(conn)
    
    engine = SentimentEngine()
    inseriti = 0
    skippati = 0
    
    # Usiamo il generatore a blocchi
    chunks_gen = NewsFetcher.load_chunks_from_kaggle_csv(csv_path, target_tickers, chunk_size=chunk_size)
    
    # Poiché non sappiamo il numero totale di righe senza leggerlo tutto, 
    # usiamo una barra di progresso che conta quanti chunk abbiamo processato
    pbar = tqdm(desc="📂 Processing Chunks", unit="chunk")
    
    for news_chunk in chunks_gen:
        if not news_chunk:
            pbar.update(1)
            continue
            
        # Aggreghiamo il chunk per Date e Ticker
        dict_agg = {}
        for item in news_chunk:
            t = item['ticker']
            d = item['date']
            if t not in dict_agg: dict_agg[t] = {}
            if d not in dict_agg[t]: dict_agg[t][d] = []
            dict_agg[t][d].append(item)
            
        # Elaborazione del chunk
        for ticker, dates_dict in dict_agg.items():
            for d, items in dates_dict.items():
                # Controllo se esiste già (per permettere di riprendere in caso di stop)
                exists = conn.execute("SELECT News_Count FROM sentiment_cache WHERE Ticker=? AND Date=?", (ticker, d)).fetchone()
                if exists:
                    skippati += 1
                    continue
                
                # Calcolo Sentiment
                score, confidence, volatility = engine.compute_daily_aggregate(items)
                
                # Salvataggio
                conn.execute('''
                    INSERT INTO sentiment_cache (Date, Ticker, Sentiment_Score, Confidence, Volatility, News_Count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (d, ticker, float(score), float(confidence), float(volatility), len(items)))
                inseriti += 1
                
        conn.commit() # Commit dopo ogni chunk per sicurezza
        pbar.update(1)
        pbar.set_postfix({"Inseriti": inseriti, "Skippati": skippati})

    conn.close()
    print(f"\n" + "=" * 60)
    print(f"🎉 IMPORTAZIONE COMPLETATA!")
    print(f"   Record elaborati e inseriti: {inseriti}")
    print(f"   Record già presenti (saltati): {skippati}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill DB con dataset Kaggle (V7 Triple Brain)")
    parser.add_argument("--csv", type=str, required=True, help="Percorso del CSV")
    parser.add_argument("--all-tickers", action="store_true", help="Non filtrare ticker (tratta tutto come GLOBAL se non a target)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Dimensione del blocco di lettura (default 50k)")
    args = parser.parse_args()
    
    # Se passiamo target_tickers=None, il caricatore cercherà comunque di mappare a target se li trova
    # ma in questo schema, passeremo sempre TARGET_TICKERS per differenziare tra Core e Global
    backfill_from_kaggle(args.csv, target_tickers=TARGET_TICKERS_AZIONARIO, chunk_size=args.chunk_size)
