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

    @staticmethod
    def get_cached_intraday_data(ticker, conn, api_key, secret_key, start_date="2019-01-01"):
        """Scarica e aggiorna i dati a 1 minuto tramite Alpaca su DB SQLite.
        
        Utilizza la tabella `intraday_cache` con risoluzione al minuto.
        I dati vengono scaricati in batch mensili per gestire la paginazione.
        """
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        conn.execute('''CREATE TABLE IF NOT EXISTS intraday_cache (
            Datetime TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL,
            Close REAL, Volume REAL, VWAP REAL,
            PRIMARY KEY (Datetime, Ticker))''')

        # Determina da dove riprendere
        res = conn.execute("SELECT MAX(Datetime) FROM intraday_cache WHERE Ticker=?", (ticker,)).fetchone()
        last_dt_str = res[0] if res else None

        if last_dt_str:
            last_dt = pd.to_datetime(last_dt_str)
            download_start = last_dt + datetime.timedelta(minutes=1)
        else:
            download_start = pd.to_datetime(start_date).tz_localize('America/New_York')

        # Fine download = oggi - 15 min (delay dati free)
        now_et = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4)))
        download_end = now_et - datetime.timedelta(minutes=16)

        if download_start.tzinfo is None:
            download_start = download_start.tz_localize('America/New_York')
        if download_end.tzinfo is None:
            download_end = download_end.tz_localize('America/New_York')

        if download_start >= download_end:
            # Già aggiornato
            return pd.read_sql_query(
                "SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker=? ORDER BY Datetime ASC",
                conn, params=(ticker,)
            )

        client = StockHistoricalDataClient(api_key, secret_key)

        # Scarica in batch mensili per evitare timeout e gestire la mole di dati
        batch_start = download_start
        total_inserted = 0
        total_batches = (download_end - download_start).days // 30 + 1
        
        from tqdm import tqdm as tqdm_bar
        pbar = tqdm_bar(total=total_batches, desc=f"  📥 {ticker}", leave=False)

        while batch_start < download_end:
            batch_end = min(batch_start + datetime.timedelta(days=30), download_end)

            for attempt in range(5):
                try:
                    request_params = StockBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=TimeFrame.Minute,
                        start=batch_start,
                        end=batch_end
                    )
                    bars = client.get_stock_bars(request_params)
                    df_bars = bars.df

                    if not df_bars.empty:
                        if isinstance(df_bars.index, pd.MultiIndex):
                            df_bars = df_bars.reset_index(level=0, drop=True)
                        df_bars = df_bars.reset_index()

                        # Normalizza il nome della colonna datetime
                        dt_col = [c for c in df_bars.columns if 'timestamp' in c.lower() or 'datetime' in c.lower()]
                        if dt_col:
                            df_bars.rename(columns={dt_col[0]: 'Datetime'}, inplace=True)
                        elif 'index' in df_bars.columns:
                            df_bars.rename(columns={'index': 'Datetime'}, inplace=True)

                        df_bars['Datetime'] = pd.to_datetime(df_bars['Datetime']).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M:%S')
                        df_bars['Ticker'] = ticker

                        col_map = {}
                        for c in df_bars.columns:
                            cl = c.lower()
                            if cl == 'open': col_map[c] = 'Open'
                            elif cl == 'high': col_map[c] = 'High'
                            elif cl == 'low': col_map[c] = 'Low'
                            elif cl == 'close': col_map[c] = 'Close'
                            elif cl == 'volume': col_map[c] = 'Volume'
                            elif cl == 'vwap': col_map[c] = 'VWAP'
                        df_bars.rename(columns=col_map, inplace=True)

                        cols_save = ['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
                        if 'VWAP' in df_bars.columns:
                            cols_save.append('VWAP')
                        else:
                            df_bars['VWAP'] = 0.0
                            cols_save.append('VWAP')

                        df_save = df_bars[cols_save].dropna(subset=['Close'])
                        if last_dt_str:
                            df_save = df_save[df_save['Datetime'] > last_dt_str]

                        if not df_save.empty:
                            df_save.to_sql('intraday_cache', conn, if_exists='append', index=False)
                            conn.commit()
                            total_inserted += len(df_save)

                    time.sleep(random.uniform(0.3, 0.8))
                    break
                except Exception as e:
                    if 'rate limit' in str(e).lower() or '429' in str(e):
                        time.sleep(60)
                    else:
                        time.sleep((attempt + 1) * 3)
                    if attempt == 4:
                        print(f"⚠️ Errore persistente per {ticker} ({batch_start.date()}→{batch_end.date()}): {e}")

            batch_start = batch_end
            pbar.update(1)

        pbar.close()
        return pd.read_sql_query(
            "SELECT Datetime, Open, High, Low, Close, Volume, VWAP FROM intraday_cache WHERE Ticker=? ORDER BY Datetime ASC",
            conn, params=(ticker,)
        )

    @staticmethod
    def resample_to_5min(df):
        """Resampla barre a 1 minuto in barre a 5 minuti.
        
        Aggregazione OHLCV corretta con rispetto dei confini di sessione.
        Il resampling avviene per date separate per non mescolare sessioni diverse.
        
        Args:
            df: DataFrame con colonne Datetime, Open, High, Low, Close, Volume, VWAP
            
        Returns:
            DataFrame a 5 minuti con le stesse colonne, ordinato cronologicamente.
        """
        df = df.copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        # Raggruppa per sessione (data) per non mescolare giorni
        df['_date'] = df['Datetime'].dt.date
        
        resampled_parts = []
        for _, day_df in df.groupby('_date'):
            if len(day_df) < 5:
                continue
            
            day_df = day_df.set_index('Datetime')
            
            agg = day_df.resample('5min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'VWAP': 'mean'  # approssimazione accettabile bar-by-bar
            }).dropna(subset=['Close'])
            
            # VWAP più preciso: ricalcola come media pesata per volume
            vol = day_df['Volume'].resample('5min').sum()
            vwap_num = (day_df['VWAP'] * day_df['Volume']).resample('5min').sum()
            precise_vwap = vwap_num / (vol + 1e-9)
            agg['VWAP'] = precise_vwap.where(vol > 0, agg['VWAP'])
            
            agg = agg.reset_index()
            agg.rename(columns={'index': 'Datetime'} if 'index' in agg.columns else {}, inplace=True)
            if 'Datetime' not in agg.columns and agg.index.name == 'Datetime':
                agg = agg.reset_index()
            
            resampled_parts.append(agg)
        
        if not resampled_parts:
            return pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])
        
        result = pd.concat(resampled_parts, ignore_index=True)
        result['Datetime'] = result['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Rimuovi colonna ausiliaria se presente
        if '_date' in result.columns:
            result.drop(columns=['_date'], inplace=True)
        
        return result.reset_index(drop=True)
