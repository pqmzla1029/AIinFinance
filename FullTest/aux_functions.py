import datetime as dt
import warnings
import numpy as np
import pandas as pd
import import_functions as ifunc

def MACD(stockdf):
    df = stockdf
    #df = df.reset_index()
    df['30 mavg'] = df['Close'].rolling(30).mean()
    df['26 ema'] = df['Close'].ewm(span = 26).mean()
    df['12 ema'] = df['Close'].ewm(span = 12).mean()
    df['MACD'] = (df['12 ema'] - df['26 ema'])
    df['Signal'] = df['MACD'].ewm(span = 9).mean()
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
    roll_up_ewma = up.ewm(span = window_length).mean()
    roll_down_ewma = down.ewm(span = window_length).mean()
    RS_ewma = roll_up_ewma / roll_down_ewma
    RSI_ewma = 100.0 - (100.0 / (1.0 + RS_ewma))

    # Calculate RSI based on SMA
    roll_up_sma = up.rolling(window_length).mean()
    roll_down_sma = down.rolling(window_length).mean()
    RS_sma = roll_up_sma / roll_down_sma
    RSI_sma = 100.0 - (100.0 / (1.0 + RS_sma))

    df['RSI SMA'] = RSI_sma
    df['RSI EWMA'] = RSI_ewma
    return df

def DELTA(stockdf):
    df = stockdf
    #df = df.reset_index()
    df['Daily Change'] = df['Close'].pct_change()
    df['5 Day Change'] = df['Close'].pct_change(periods = 5)
    return df


def main():
    filename,date1,date2= ifunc.read_file()
    filename=filename+".csv"
    #print(filename)
    stockdf = ifunc.read_full(filename)
    stockdf = MACD(stockdf)
    stockdf = RSI(stockdf)
    stockdf = DELTA(stockdf)
    stockdf.to_csv(path_or_buf=filename,index=False)

main()
