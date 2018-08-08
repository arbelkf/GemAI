# Auther : Kfir Arbel
# date : 6.8.2018
# Common Class

import datetime as dt

start = dt.datetime(1995, 1, 1)
end = dt.datetime(2018, 7, 11)
filePathProcessedStocks = "ProcessedStocks"
filePathIndexes = "Indexes"
filePathStocks = "stock_dfs"

indexesList = ['^DJI','^GDAXI','^HSI','^FCHI','^GSPC','^IXIC', '^N225','^RUT', '^TYX']

'''indicators = ['volume_delta', 'open_-2_r', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'volume_-3,2,-1_max', 'volume_-3~1_min',
              'kdjk', 'kdjd', 'kdjj', 'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb',
              'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 'wr_10', 'wr_6', 'cci', 'cci_20', 'tr',
              'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'adxr', 'trix', 'trix_9_sma', 'vr', 'vr_6_sma']'''

'''indicators = [ 'atr','boll_ub', 'boll','cr-ma3','boll_lb', 'open_2_sma', 'vr_6_sma','cr-ma2', 'adxr', 'volume_-3~1_min','vr','cr-ma1','trix_9_sma','volume_-3,2,-1_max',
                    'kdjd','macdh', 'adx']'''
indicators = [ 'atr','boll_ub', 'cr-ma3', 'open_2_sma', 'vr_6_sma', 'adxr', 'volume_-3~1_min','trix_9_sma','volume_-3,2,-1_max', 'kdjd','macdh', 'adx']