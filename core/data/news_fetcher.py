import requests
import datetime
import os
import yfinance as yf
from core.config import FINNHUB_API_KEY, MARKETAUX_API_KEY, ALPHA_VANTAGE_API_KEY, NEWSAPI_KEY
import time

class NewsFetcher:
    """Fetcher multifonte per estrarre le notizie."""
    
    @staticmethod
    def get_finnhub_news(ticker, start_date, end_date):
        """Usa l'API di Finnhub (se disponibile) per estrarre news."""
        if not FINNHUB_API_KEY:
            return []
            
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                news = []
                for item in data:
                    t = datetime.datetime.fromtimestamp(item.get('datetime', 0))
                    news.append({
                        'date': t.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'title': item.get('headline', ''),
                        'summary': item.get('summary', '')
                    })
                return news
        except Exception as e:
            print(f"Errore Finnhub: {e}")
        return []

    @staticmethod
    def get_yfinance_news(ticker):
        """Usa YFinance (gratuito ma limitato ai giorni recenti) per news recenti."""
        news = []
        try:
            tkr = yf.Ticker(ticker)
            yf_news = tkr.news
            
            for item in yf_news:
                # yfinance returns providerPublishTime
                pub_time = item.get('providerPublishTime', 0)
                t = datetime.datetime.fromtimestamp(pub_time)
                news.append({
                    'date': t.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'title': item.get('title', ''),
                    'summary': '' # yfinance non fornisce sempre una summary nei feed base
                })
        except Exception as e:
            print(f"Errore YFinance: {e}")
        return news

    @staticmethod
    def get_alphavantage_news(ticker, limit=50):
        """Usa l'API Alpha Vantage per estrarre news (limitata per chiavi free)."""
        if not ALPHA_VANTAGE_API_KEY:
            return []
            
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "feed" not in data:
                    return []
                    
                news = []
                for item in data['feed']:
                    # Il formato data Alpha Vantage è: 20240320T173000
                    time_str = item.get('time_published', '')
                    if len(time_str) >= 8:
                        date_str = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}"
                        news.append({
                            'date': date_str,
                            'ticker': ticker,
                            'title': item.get('title', ''),
                            'summary': item.get('summary', '')
                        })
                return news
        except Exception as e:
            print(f"Errore Alpha Vantage: {e}")
        return []

    @staticmethod
    def get_newsapi_org_news(ticker, start_date, end_date):
        """Usa NewsAPI.org per estrarre news (limitata per i free tier a 30 giorni indietro, ma molto utile)."""
        if not NEWSAPI_KEY:
             return []
             
        # Le ricerche generiche per stock a volte riportano rumore, aggiungiamo "stock" o usiamo nomi di azienda noti
        ticker_search_map = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 'AMZN': 'Amazon', 
            'NVDA': 'Nvidia', 'TSLA': 'Tesla', 'META': 'Meta Platforms'
        }
        
        query = ticker_search_map.get(ticker, f"{ticker} stock")
        url = f"https://newsapi.org/v2/everything?q={query}&from={start_date}&to={end_date}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "articles" not in data: return []
                
                news = []
                for item in data['articles']:
                    pub_str = item.get('publishedAt', '')
                    if len(pub_str) >= 10:
                        Date_str = pub_str[:10]
                        news.append({
                            'date': Date_str,
                            'ticker': ticker,
                            'title': item.get('title', ''),
                            'summary': item.get('description', '')
                        })
                return news
        except Exception as e:
            print(f"Errore NewsAPI: {e}")
        return []

    @staticmethod
    def get_recent_news(ticker, force_all=False, days_back=7):
        """
        Recupera le notizie recenti combinando i feed.
        """
        news_list = []
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # 1. Prova YFinance (sempre free, veloce ma limitato in quantità)
        yf_n = NewsFetcher.get_yfinance_news(ticker)
        news_list.extend(yf_n)
        
        # 2. Prova Finnhub (se la chiave esiste)
        if FINNHUB_API_KEY:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            fh_n = NewsFetcher.get_finnhub_news(ticker, start_date, end_date)
            news_list.extend(fh_n)
            time.sleep(1.5) # rispetta i limiti finnhub free (60 req / min)

        # 3. Prova Alpha Vantage (se chiave esiste e force_all è true)
        if ALPHA_VANTAGE_API_KEY and force_all:
             av_n = NewsFetcher.get_alphavantage_news(ticker, limit=50 if days_back > 7 else 20)
             news_list.extend(av_n)
             time.sleep(1)
             
        # 4. Prova NewsAPI
        if NEWSAPI_KEY and force_all:
             na_n = NewsFetcher.get_newsapi_org_news(ticker, start_date, end_date)
             news_list.extend(na_n)
             time.sleep(1)
             
        # Dedup: Le news vengono duplicate spesso dalle stesse fonti. Usiamo il titolo come chiave.
        unique_news = {}
        for n in news_list:
            if n['title'] and n['title'] not in unique_news:
                unique_news[n['title']] = n
                
        return list(unique_news.values())

    @staticmethod
    @staticmethod
    def load_chunks_from_kaggle_csv(csv_path, target_tickers=None, chunk_size=50000):
        """
        Generatore che carica un file CSV a pezzi (chunks) per risparmiare RAM.
        Implementa la logica GLOBAL per i ticker non a target.
        """
        import pandas as pd
        if not os.path.exists(csv_path):
            print(f"File non trovato: {csv_path}")
            return
            
        try:
             # Analisi headers
             df_sample = pd.read_csv(csv_path, nrows=1)
             columns = df_sample.columns.tolist()
             
             # Determina il formato e le colonne da caricare
             if 'datetime' in columns and 'headline' in columns and 'symbol' not in columns:
                  loader_type = 'google'
                  use_cols = ['headline', 'datetime']
             elif 'symbol' in columns and 'headline' in columns and 'date' in columns:
                  loader_type = 'symbol_standard'
                  use_cols = ['headline', 'date', 'symbol']
             else:
                  loader_type = 'stock_standard'
                  use_cols = ['headline', 'date', 'stock']
                  
             # Iteratore sui chunks
             reader = pd.read_csv(csv_path, usecols=use_cols, chunksize=chunk_size)
             
             for chunk in reader:
                  news_chunk = []
                  
                  if loader_type == 'google':
                       chunk['date'] = pd.to_datetime(chunk['datetime'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d')
                       chunk = chunk.dropna(subset=['date', 'headline'])
                       for _, row in chunk.iterrows():
                           news_chunk.append({'date': row['date'], 'ticker': 'GOOGL', 'title': row['headline'], 'summary': ''})
                           
                  elif loader_type == 'symbol_standard':
                       # Millisecondi o stringa?
                       if chunk['date'].dtype == 'int64' or chunk['date'].dtype == 'float64':
                           chunk['date'] = pd.to_datetime(chunk['date'], unit='ms', errors='coerce').dt.strftime('%Y-%m-%d')
                       else:
                           chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                       
                       chunk = chunk.dropna(subset=['date', 'headline'])
                       for _, row in chunk.iterrows():
                           ticker = str(row['symbol']).strip()
                           if target_tickers and ticker not in target_tickers:
                                ticker = 'GLOBAL'
                           news_chunk.append({'date': row['date'], 'ticker': ticker, 'title': row['headline'], 'summary': ''})
                           
                  else: # stock_standard
                       chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                       chunk = chunk.dropna(subset=['date', 'headline'])
                       for _, row in chunk.iterrows():
                           ticker = str(row['stock']).strip()
                           if target_tickers and ticker not in target_tickers:
                                ticker = 'GLOBAL'
                           news_chunk.append({'date': row['date'], 'ticker': ticker, 'title': row['headline'], 'summary': ''})
                           
                  yield news_chunk

        except Exception as e:
             print(f"Errore caricamento chunks CSV: {e}")
             return
