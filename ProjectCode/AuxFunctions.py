import datetime as dt
import warnings
import numpy as np
import pandas as pd

def MACD(stockdf):
    df = stockdf
    df = df.reset_index()
    df['30 mavg'] = pd.rolling_mean(df['Close'],30)
    df['26 ema'] = pd.ewma(df['Close'], span = 26)
    df['12 ema'] = pd.ewma(df['Close'], span = 12)
    df['MACD'] = (df['12 ema'] - df['26 ema'])
    df['Signal'] = pd.ewma(df['MACD'], span = 9)
    df['Crossover'] = df['MACD'] - df['Signal']
    return df

def RSI(stockdf):
    df = stockdf
    window_length = 10
    close = df['Close']
    delta = close.diff()

    up, down = delta.copy(), delta.copy()
    up[up<0] = 0
    down[down>0] = 0

    # Calculate RSI based on EWMA
    roll_up_ewma = pd.ewma(up, window_length)
    roll_down_ewma = pd.ewma(down, window_length)
    RS_ewma = roll_up_ewma / roll_down_ewma
    RSI_ewma = 100.0 - (100 / (1.0 + RS_ewma))

    # Calculate RSI based on SMA
    roll_up_sma = pd.rolling_mean(up, window_length)
    roll_down_sma = pd.rolling_mean(down, window_length)
    RS_sma = roll_up_sma / roll_down_sma
    RSI_sma = 100.0 - (100.0 / (1.0 + RS_sma))

    df['RSI SMA'] = RSI_sma
    df['RSI EWMA'] = RSI_ewma
    return df


def deltaPercentage(stockdf):

