# Auther : Kfir Arbel
# date : 6.8.2018
# predict for spikes class
# looks for spike(major change in the value in a short time)
# the percent change is calculated from a day to the one following it.

from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from GemAIModule.DataFrameUtils import DataFrameUtils
from sklearn import preprocessing
from StrategyModule.SpikeForPeriodStrategy import SpikeForPeriodStrategy
from StrategyModule.Context import Context
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



class PredictSpikes(object):

    def __init__(self):
        print("Start...")

        # list_of_params = [(0.05, 0.05, 0.1, 0.1, 14), (0.05, 0.05, 0.15, 0.15, 14),
        #                   (0.03, 0.03, 0.1, 0.1, 14), (0.03, 0.03, 0.15, 0.15, 14),
        #                   (0.05, 0.05, 0.1, 0.1, 5), (0.05, 0.05, 0.15, 0.15, 5),
        #                   (0.03, 0.03, 0.1, 0.1, 5), (0.03, 0.03, 0.15, 0.15, 5),
        #                   (0.05, 0.05, 0.05, 0.05, 8), (0.04, 0.04, 0.07, 0.07, 8),
        #                   (0.05, 0.05, 0.05, 0.05, 5), (0.04, 0.04, 0.07, 0.07, 5),
        #                   (0.03, 0.03, 0.05, 0.05, 3), (0.03, 0.03, 0.10, 0.10, 3),
        #                   (0.03, 0.03, 0.05, 0.05, 2), (0.03, 0.03, 0.10, 0.10, 2)
        #                   ]

        list_of_params = [(0.05, 0.05, 0.1 ,0.1, 7)]


        self._StrategyList = []
        for item in list_of_params:
            str = SpikeForPeriodStrategy(name="SpikeForPeriodStrategy",
                                         spikeLowest=item[0] ,spikeHighest=item[1], highestLimit=item[2], lowestLimit=item[3],  hm_days=item[4],
                                                    filePathIndexes="..\\ImportModule\\Indexes\\")

            self._StrategyList.append(str)



    def DoStrategy(self, strategy, ticker, skipPredict = False):
        context = Context(strategy)
        acc, confusionmatrix, final = context.ProcessTicker("stock_ndx_processed\\processed_", ticker, skipPredict)
        return acc, confusionmatrix, final

    def PredictTicker(self, ticker):

        try:
            print("Processing:{}".format(ticker))

            df = pd.DataFrame()
            labels = ['name', 'acc', 'BuyPercent', 'spikelow' , 'spikehigh', 'highest', 'lowest', 'Hm_days', '_-1 ', '_0 ', '_1 ', '_(0,1)',
                      '_(0,2) ', '_(1,0)', '_(1,2)', '_(2,0)', '_(2,1)']

            for item in self._StrategyList:
                acc, confusionmatrix, final = self.DoStrategy(item, ticker, skipPredict=True)
                #newLine = np.array([[confusionmatrix[1,1]]])
                if (acc == None):
                    return None
                x,y = confusionmatrix.shape
                confusum = confusionmatrix[0, 2] +  confusionmatrix[2, 0] + confusionmatrix[2, 2]
                BuyPercentAcc = 1
                if (confusum != 0):
                    BuyPercentAcc = confusionmatrix[2, 2] / (
                            confusionmatrix[0, 2] +  confusionmatrix[2, 0] + confusionmatrix[2, 2])
                else:
                    if (confusionmatrix[2, 2] == 0):
                        BuyPercentAcc = 0
                if (x > 2):
                    c = np.array(
                        [[item.Name , acc, BuyPercentAcc,item._spikeLowest, item._spikeHighest, item.HighestLimit, item.LowestLimit, item.Hm_days, confusionmatrix[0, 0],
                          confusionmatrix[1, 1], confusionmatrix[2, 2], confusionmatrix[0, 1], confusionmatrix[0, 2],
                          confusionmatrix[1, 0],
                          confusionmatrix[1, 2], confusionmatrix[2,0], confusionmatrix[2, 1]]])
                    if (len(df) > 0 ):
                        df = df.append(pd.DataFrame(c, columns=labels))
                    else:
                        df = pd.DataFrame(c,  columns=labels)
                else:
                    print("Skipped...")
            filename = "spikestrategyparams_processed\\results_{}_{}.xlsx".format(item.Name, ticker)

            writer = pd.ExcelWriter(filename)
            df.to_excel(writer, 'Sheet1')
            writer.save()


            #df.to_csv(filename)


        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

    def PredictAll(self, filepath = '..\ImportModule\\ndx.csv'):
        data = pd.read_csv(filepath)
        tickers = data['ticker']

        for ticker in tickers[7:8]:
            sys.stdout.write('.')
            self.PredictTicker(ticker)

pred = PredictSpikes()
pred.PredictAll()
print("END")


