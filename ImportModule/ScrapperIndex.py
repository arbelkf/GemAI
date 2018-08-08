# Auther : Kfir Arbel
# date : 6.8.2018
# ScrapperIndex class
# retreives all indexes data from the web


import pandas_datareader as web
from pandas_datareader import data as pdr

import os,sys,inspect
import os.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import Utils.Common as common

import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)


def ReloadIndex(ticker):
    filepath = common.filePathIndexes
    if not os.path.exists(filepath):
            os.makedirs(filepath)
    filename = common.filePathIndexes + '\{}.csv'.format(ticker)
    if (os.path.isfile(filename)):
        print('skipping {}'.format(ticker))
        return
    else:
        print('downloading {}'.format(ticker))
    try:
        #df2 = pdr.get_data_yahoo(ticker, start=common.start, end=common.end)
        df2 = web.DataReader(ticker, 'yahoo', common.start, common.end)
        df2.to_csv(filepath + '\{}.csv'.format(ticker))
        return
    except ValueError:
        print("Oops!  yahoo.  Try again...", ticker, "   " )

    try:
        df2 = web.DataReader(ticker, 'google', common.start, common.end)
        df2.to_csv(filepath + '\{}.csv'.format(ticker))
        return
    except ValueError:
        print("Oops!  google.  Try again...", ticker, "   " ,ValueError )

    try:
        df2 = web.DataReader(ticker, 'quandl', common.start, common.end)
        df2.to_csv(filepath + '\{}.csv'.format(ticker))
        return
    except ValueError:
        print("Oops!  quandl.  Try again...", ticker, "   " ,ValueError )

    try:
        df2 = web.DataReader(ticker, 'fred', common.start, common.end)
        df2.to_csv(filepath + '\{}.csv'.format(ticker))
        return
    except ValueError:
        print("Oops!  fred.  Try again...", ticker, "   " ,ValueError )


def reloadIndexes():
    for index in common.indexesList:
        ReloadIndex(index)

reloadIndexes()
print('End')