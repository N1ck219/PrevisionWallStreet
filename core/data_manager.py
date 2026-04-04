import sqlite3
import pandas as pd
import datetime
import yfinance as yf
import random
import time
import os

class DataManager:
    @staticmethod
    def setup_db(db_path):
        """Crea la connessione al database e assicura che le directory esistano."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path, timeout=30, check_same_thread=False)

    @staticmethod
    def get_cached_market_data(ticker, conn, start_date="2020-01-01"):
        """Scarica e aggiorna i dati storici tramite YFinance su DB SQLite"""
        conn.execute('''CREATE TABLE IF NOT EXISTS market_cache (Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL, PRIMARY KEY (Date, Ticker))''')
        res = conn.execute("SELECT MAX(Date) FROM market_cache WHERE Ticker=?", (ticker,)).fetchone()
        last_date_str = res[0] if res else None
        
        ora_attuale = datetime.datetime.now()
        oggi = (ora_attuale - datetime.timedelta(days=1)).date() if ora_attuale.hour < 23 else ora_attuale.date()
        
        scarica = False
        if not last_date_str: 
            scarica = True
        else:
            last_date = datetime.datetime.strptime(last_date_str, '%Y-%m-%d').date()
            if last_date < oggi: 
                scarica = True
                start_date = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        if scarica:
            for attempt in range(4):
                try:
                    tkr = yf.Ticker(ticker)
                    df_new = tkr.history(start=start_date, end=(oggi + datetime.timedelta(days=1)).strftime('%Y-%m-%d')).reset_index()
                    if not df_new.empty:
                        if 'Datetime' in df_new.columns: 
                            df_new.rename(columns={'Datetime': 'Date'}, inplace=True)
                        df_new['Date'] = pd.to_datetime(df_new['Date'], utc=True).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
                        df_new['Ticker'] = ticker
                        df_save = df_new[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                        if last_date_str: 
                            df_save = df_save[df_save['Date'] > last_date_str]
                        if not df_save.empty: 
                            df_save.to_sql('market_cache', conn, if_exists='append', index=False)
                    time.sleep(random.uniform(0.2, 1.0))
                    break
                except Exception as e:
                    time.sleep((attempt + 1) * 2)
                    
        return pd.read_sql_query("SELECT Date, Open, High, Low, Close, Volume FROM market_cache WHERE Ticker=? ORDER BY Date ASC", conn, params=(ticker,))
