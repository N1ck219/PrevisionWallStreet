import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import torch
from transformers import pipeline

class SentimentEngine:
    """Motore per l'analisi del sentiment basato su FinBERT."""
    def __init__(self, model_name="ProsusAI/finbert"):
        # Controlla se c'è una GPU disponibile
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"L'analizzatore di sentiment utilizza: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")
        
        # Caricamento del modello FinBERT con batch_size per abilitare calcolo in parallelo su GPU
        self.analyzer = pipeline("sentiment-analysis", model=model_name, device=self.device, batch_size=32)

    def analyze_batch(self, texts):
        """
        Analizza un batch di testi.
        Ritorna una lista di dizionari: {'label': 'positive|negative|neutral', 'score': float}
        """
        if not texts:
            return []
        
        # Tronca i testi molto lunghi (solitamente FinBERT supporta fino a 512 token)
        # Limitiamo il numero di caratteri per sicurezza, assumendo che i primi 500 coprano l'essenziale
        truncated_texts = [text[:2000] for text in texts]
        
        try:
            # Passiamo un set limitato di testi e lasciamo che la pipeline gestisca i batch (batch_size=32 def.)
            results = self.analyzer(truncated_texts)
            return results
        except Exception as e:
            print(f"Errore durante l'analisi del batch: {e}")
            # Ritorna neutrale in caso di errore
            return [{'label': 'neutral', 'score': 0.0} for _ in texts]

    def compute_daily_aggregate(self, news_items):
        """
        Calcola i punteggi aggregati per una lista di news in un determinato giorno.
        Ritorna:
        - sentiment_score_netto (float [-1, 1]): Positivo - Negativo pesato.
        - confidence_media (float): Indice di certezza.
        - sentiment_volatility (float): Indice di volatilità del sentiment.
        """
        if not news_items:
            return 0.0, 0.0, 0.0
        # Estrai i testi e concatenali. Assicurati che titolo e summary non siano None
        texts = [(item.get('title') or '') + ". " + (item.get('summary') or '') for item in news_items]
        results = self.analyze_batch(texts)
        
        # Assegna valenze: positivo = 1, negativo = -1, neutrale = 0
        scores = []
        confidences = []
        
        for res in results:
            valenza = 0
            if res['label'] == 'positive':
                valenza = 1
            elif res['label'] == 'negative':
                valenza = -1
                
            # Il sentiment netto è valenza * livello di confidenza
            scores.append(valenza * res['score'])
            confidences.append(res['score'])
            
        import numpy as np
        
        # Score_netto [-1.0, 1.0]
        sentiment_score_netto = np.mean(scores)
        
        # Se tutti score sono concordi e alti -> confidence media.
        confidence_media = np.mean(confidences)
        
        # Misura se ci sono tante buone e cattive notizie simultaneamente (voto per l'incertezza del mercato)
        sentiment_volatility = np.std(scores) if len(scores) > 1 else 0.0
        
        return sentiment_score_netto, confidence_media, sentiment_volatility
