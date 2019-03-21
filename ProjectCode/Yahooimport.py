#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:42:05 2019

@author: thurteen
"""

import datetime
import warnings
import numpy as np
import fix_yahoo_finance as yf
import pandas as pd
#pd.core.common.is_list_like = pd.api.types.is_list_like
#import pandas_datareader as pdr

def retrieve_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    return data
    #print(data)
    
    
    
#retrieve_data("AAPL", "2012-01-01", "2019-01-01")