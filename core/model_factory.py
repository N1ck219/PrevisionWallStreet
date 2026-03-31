import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, concatenate
from tensorflow.keras.optimizers import Adam

def build_v4_3_model(input_shape):
    """Architettura per la versione 4.3"""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True, name="lstm_base")(inputs)
    x = Dropout(0.2)(x)
    att = Attention(name="att")([x, x])
    x = LSTM(64, name="lstm_feat")(att)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    return model

def build_v4_6_model(input_shape):
    """Architettura profonda per le versioni 4.6 e Crypto 1.7"""
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True, name="lstm_base")(inputs)
    x = Dropout(0.3)(x)
    att = Attention(name="att")([x, x])
    x = LSTM(64, name="lstm_feat")(att)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy')
    return model

def build_v5_0_split_brain_model(shape_t, shape_m):
    """Architettura a doppio input (Split Brain) per V5.6 e V6.4"""
    in_t = Input(shape=shape_t)
    x_t = LSTM(128, return_sequences=True)(in_t)
    x_t = Dropout(0.3)(x_t)
    att = Attention()([x_t, x_t])
    x_t = LSTM(64)(att)
    
    in_m = Input(shape=shape_m)
    x_m = LSTM(64)(in_m) 
    
    merged = concatenate([x_t, x_m])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[in_t, in_m], outputs=out)
    return model

def get_model(version, weights_path=None, input_shape=None, shape_t=None, shape_m=None):
    """
    Funzione universale per istanziare e caricare i pesi del modello richiesto.
    """
    if version == "4.3":
        model = build_v4_3_model(input_shape)
    elif version in ["4.6", "crypto_1.7"]:
        model = build_v4_6_model(input_shape)
    elif version in ["5.6", "6.4"]:
        model = build_v5_0_split_brain_model(shape_t, shape_m)
    else:
        raise ValueError(f"Versione modello '{version}' non riconosciuta.")
        
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        
    return model