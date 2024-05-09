import os
import torch
import datetime
import numpy as np
import pandas as pd
import math

import talib as ta
import qlib
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.constant import REG_CN, REG_US

provider_uri = "F:/qlib/qlib_data/us_data" # data dir
qlib.init(provider_uri=provider_uri, region=REG_US)

def get_base_company(start_time, end_time):
    instruments = D.instruments(market='nasdaq100')
    company_pr = D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True)
    company_pr.sort()
    return company_pr

def get_data(start_time, end_time, selected_tickers, market='nasdaq100'):
    if selected_tickers is None:
        instruments = D.instruments(market=market)
        all_tickers = (D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True))
        selected_tickers = all_tickers
    else:
        instruments = selected_tickers
    window1, window2, window3 = 5, 20, 60
    # get timestamps
    all_timestamps = list(D.calendar(start_time=start_time, end_time=end_time))
    
    # get data
    all_data = pd.read_csv('./dataset/all_data.csv')
    all_data = all_data.dropna()

    for timestamp in all_timestamps:
        tickers = list(all_data.loc[timestamp].index)
        union = list(set(tickers) & set(selected_tickers))
        selected_tickers = union

    selected_tickers.sort(reverse=False)
    selected_data = all_data.loc[(slice(None), selected_tickers), :]

    # examine alignment
    if len(selected_data) == len(all_timestamps) * len(selected_tickers):
        return all_timestamps, selected_tickers, selected_data
    else:
        raise Exception('Data is not aligned.')
    
def ZscoreNorm(series):
    return (series-np.mean(series))/np.std(series)

def add_alpha(args, selected_tickers):
    print('Loading base technical data...')
    all_timestamps, all_tickers, all_data = get_data(
        start_time=args.prestart_time, 
        end_time=args.lagend_time, 
        selected_tickers=selected_tickers, 
        market='sp500',
        )
    data_with_alpha = all_data.copy()
    
    print('Loading indicators...')
    for comp in all_tickers:
        close_series = all_data.loc[:,'feature'].loc[:,'$close'].swaplevel().loc[comp, :]
        high_series = all_data.loc[:,'feature'].loc[:,'$high'].swaplevel().loc[comp, :]
        low_series = all_data.loc[:,'feature'].loc[:,'$low'].swaplevel().loc[comp, :]
        volume_series = all_data.loc[:,'feature'].loc[:,'$volume'].swaplevel().loc[comp, :]
        return_series = all_data.loc[:,'feature'].loc[:,'$close/Ref($close, 1)-1'].swaplevel().loc[comp, :]

        df_alpha = pd.DataFrame(close_series)
        types_all = ['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','T3',
                    'atr', 'natr', 'trange',
                    'rsi',
                    'obv', 'norm_return', 'obv-ref',
                    'macd', 'macdsignal', 'macdhist', 'macdhist-ref',
                    'slowk', 'slowd', 'norm_kdjhist', 'kdjhist-ref',]
        
        # MA
        types_ma=['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','T3']
        for i in range(len(types_ma)):
            df_alpha[types_ma[i]] = ZscoreNorm(ta.MA(close_series, timeperiod=5, matype=i))
        # Volatility Indicators
        df_alpha['atr'] = ZscoreNorm(ta.ATR(high_series, low_series, close_series, timeperiod=5))
        df_alpha['natr'] = ZscoreNorm(ta.NATR(high_series, low_series, close_series, timeperiod=5))
        df_alpha['trange'] = ZscoreNorm(ta.TRANGE(high_series, low_series, close_series))
        # RSI
        df_alpha['rsi'] = ta.RSI(close_series, timeperiod=5) / 100


        # Volume Indicators
        df_alpha['obv'] = ZscoreNorm(ta.OBV(close_series, volume_series))
        df_alpha['norm_return'] = ZscoreNorm(return_series)
        df_alpha['obv-ref'] = df_alpha['obv'] - df_alpha['obv'].shift(1)
        # MACD
        macd, macdsignal, macdhist = ta.MACD(close_series, fastperiod=12, slowperiod=26, signalperiod=9)
        df_alpha['macd'], df_alpha['macdsignal'], df_alpha['macdhist'] = ZscoreNorm(macd), ZscoreNorm(macdsignal), ZscoreNorm(macdhist)
        df_alpha['macdhist-ref'] = ZscoreNorm(df_alpha['macdhist'] - df_alpha['macdhist'].shift(1))
        # KDJ
        slowk, slowd = ta.STOCH(high_series, low_series, close_series, fastk_period=5, slowk_period=3)
        df_alpha['slowk'] = ZscoreNorm(slowk)
        df_alpha['slowd'] = ZscoreNorm(slowd)
        df_alpha['norm_kdjhist'] = ZscoreNorm(slowk - slowd)
        df_alpha['kdjhist-ref'] = ZscoreNorm((slowk-slowd) - (slowk-slowd).shift(1))


        newindex = pd.MultiIndex.from_product([df_alpha.index.to_list(), [comp]], names=['datetime', 'instrument'])
        df_alpha.set_index(newindex, inplace=True)
        for type in types_all:
            data_with_alpha.loc[(slice(None), comp), ('alpha', type)] = df_alpha[type]
            
    data_with_alpha = data_with_alpha.loc[datetime.datetime.strptime(args.start_time, '%Y-%m-%d'):datetime.datetime.strptime(args.end_time, '%Y-%m-%d')]
    if pd.isnull(data_with_alpha.values).any():
        print('Exist Nan')

    final_timestamps = list(D.calendar(start_time=args.start_time, end_time=args.end_time))
    if len(data_with_alpha) == len(final_timestamps) * len(all_tickers):
        return final_timestamps, all_tickers, data_with_alpha
    else:
        raise Exception('Data is not aligned.')
    
def get_features_n_labels(args, selected_tickers):
    final_timestamps, all_tickers, data_with_alpha = add_alpha(args, selected_tickers=selected_tickers)

    num_times = len(final_timestamps)
    num_nodes = len(all_tickers)
    num_features_n_label = data_with_alpha.shape[1]

    raw_data = torch.Tensor(data_with_alpha.values)
    features_n_labels = raw_data.reshape(num_times, num_nodes, num_features_n_label) # time, nodes, feature
    features = features_n_labels[:, :, 1:]
    labels = features_n_labels[:, :, 0]
    return features, labels, all_tickers, final_timestamps

def get_windows(inputs, targets, dataset, device, shuffle=False):
    window_size = 12
    
    time_length = len(inputs) - window_size + 1
    if dataset == 'train':
        start = 0
        end = math.floor(time_length / 10 * 8)
    elif dataset == 'valid':
        start = math.floor(time_length / 10 * 8) + 1
        end = math.floor(time_length / 10 * 9)
    elif dataset == 'test':
        start = math.floor(time_length / 10 * 9) + 1
        end = time_length - 1
    else:
        raise Exception('Unknown dataset.')
    length = end - start + 1
    if shuffle == True:
        indexs = torch.randperm(length)
    elif shuffle == False:
        indexs = range(length)

    for index in indexs:
        i = index + start
        window = inputs[i:(i+window_size), :, :]
        x_window = window.permute(1, 0, 2).to(device).squeeze()
        y_window = targets[i+window_size-1].to(device).squeeze()

        yield x_window, y_window