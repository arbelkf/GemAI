# Auther : Kfir Arbel
# date : 6.8.2018
#  FeaturesLab Class
# experiment labs for dataframes

import numpy as np
from scipy.stats import pearsonr
import pandas as pd

from GemAIModule.DataFrameUtils import DataFrameUtils

class FeaturesLab():

    def __init__(self):
        self._featureList = ['atr', 'boll', 'boll_ub','boll_lb','open_2_sma','cr-ma3', 'volume_-3~1_min',
                                'vr_6_sma','cr-ma2','trix_9_sma','volume_-3,2,-1_max','vr',
                                'macds','adxr','cr-ma1','dma']

    # find the covariance between the diffrence features - sort it by desending order
    def FindCovs(self, filename):
        dfUtils = DataFrameUtils()

        df = dfUtils.GetFeaturesFromCSV(filename, self._featureList)
        corr = df.corr()  # .to_csv("corr.csv")
        dftest = corr[(corr.abs() > 0.8) & (corr.abs() < 1.0)]
        dftest.to_csv('corr.csv')

        flat_cm = dftest.stack().reset_index()
        flat_cm['A_vs_B'] = flat_cm.level_0 + '_' + flat_cm.level_1
        flat_cm.columns = ['A', 'B', 'correlation', 'A_vs_B']
        flat_cm = flat_cm.loc[flat_cm.correlation < 1, ['A_vs_B', 'correlation']]
        print(flat_cm)



lab = FeaturesLab()
lab.FindCovs("stock_ndx_processed\\processed_AAPL.csv")
