import pandas as pd
import numpy as np
from core.config import MACRO_LABELS_ORDERED

class FeatureEngine:
    @staticmethod
    def add_technical_indicators(df_in):
        df = df_in.copy()
        delta = df['prezzo'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        if 'High' in df.columns and 'Low' in df.columns:
            tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['prezzo'].shift()), abs(df['Low']-df['prezzo'].shift())], axis=1).max(axis=1)
            df['ATRr_14'] = tr.rolling(14).mean()
            
        df['SMA_20'] = df['prezzo'].rolling(20).mean()
        df['STD_20'] = df['prezzo'].rolling(20).std() + 1e-9
        df['Bollinger_%B'] = (df['prezzo'] - (df['SMA_20'] - 2*df['STD_20'])) / (4*df['STD_20'])
        df['Bollinger_Width'] = (4 * df['STD_20']) / (df['SMA_20'] + 1e-9)
        
        df['SMA_200'] = df['prezzo'].rolling(200).mean()
        df['Dist_SMA200'] = (df['prezzo'] - df['SMA_200']) / (df['SMA_200'] + 1e-9)
        
        df['SMA_50'] = df['prezzo'].rolling(50).mean()
        df['Dist_SMA50'] = (df['prezzo'] - df['SMA_50']) / (df['SMA_50'] + 1e-9)
        
        df['ret'] = df['prezzo'].pct_change().fillna(0)
        
        if 'Volume' in df.columns:
            df['vol_ret'] = df['Volume'].pct_change().fillna(0)
            df['OBV_ret'] = (np.sign(df['ret']) * df['Volume']).fillna(0).cumsum().pct_change().fillna(0)
            
        return df

    @staticmethod
    def process_stock_features(df_t, macro_dict=None):
        df = df_t.copy()
        if macro_dict:
            for label, df_m in macro_dict.items(): 
                df = df.merge(df_m, on='Date', how='left')
                
        df = df.ffill().bfill()
        
        if 'Close' in df.columns:
            df.rename(columns={'Date': 'data', 'Close': 'prezzo'}, inplace=True)
            
        df = FeatureEngine.add_technical_indicators(df)
        
        if macro_dict:
            # Crea feature macro per compatibilità con tutti i bot azionari (v4.3, 4.6, 5.6, 6.4)
            mapping_v4 = {'nasdaq_close': 'nasdaq_ret', 'vix_close': 'vix_ret', 'tnx_close': 'tnx_ret', 'soxx_close': 'soxx_ret', 'gld_close': 'gld_ret'}
            for old, new in mapping_v4.items():
                if old in df.columns:
                    ret_calc = df[old].pct_change().fillna(0)
                    df[new] = ret_calc
                    df[f"{old}_ret"] = ret_calc  # Usato da v5.6 e v6.4
                    
        return df.replace([np.inf, -np.inf], np.nan).fillna(0).dropna()
    
    @staticmethod
    def extract_features(df, macro_labels=MACRO_LABELS_ORDERED):
        # Questo serve per la V5.6 e V6.4 che estraggono tensori separati
        tech_cols = ['ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA200', 'OBV_ret']
        macro_cols = [f"{m}_ret" for m in macro_labels]
        return df[tech_cols].values, df[macro_cols].values

    @staticmethod
    def process_crypto_features(df_raw):
        """Feature engineering per crypto (senza dati macro, usa SMA_50 invece di SMA_200)."""
        df = df_raw.copy()
        df = FeatureEngine.add_technical_indicators(df)
        return df.replace([np.inf, -np.inf], np.nan).fillna(0).dropna()
