# Auther : Kfir Arbel
# date : 6.8.2018
# Predict Class
# predict the future decision about a stock using its own data and diffrent params
# BuyPercentAcc = confusionmatrix[2, 2] / confusionmatrix[0, 2] +  confusionmatrix[2, 0] + confusionmatrix[2, 2]
# BuyPercentAccAqurate = confusionmatrix[2, 2]  / confusionmatrix[0, 2] + confusionmatrix[2, 0] + confusionmatrix[0, 1] + confusionmatrix[1, 0] + confusionmatrix[2, 2]

from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from GemAIModule.DataFrameUtils import DataFrameUtils
from sklearn import preprocessing
from StrategyModule.PercentForPeriodStrategy import PercentForPeriodStrategy
from StrategyModule.PercentForPeriodIndexCorrStrategy import PercentForPeriodIndexCorrStrategy
from StrategyModule.PercentForPeriodIndexLogCorrStrategy import PercentForPeriodIndexLogCorrStrategy
from StrategyModule.PercentForPeriodIndexCorrRollingStrategy import PercentForPeriodIndexCorrRollingStrategy
from StrategyModule.Context import Context
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



class Predict(object):

    def __init__(self):
        print("Start...")
        self._df = pd.DataFrame()

        # parameters for the highest value changed from the base value/lowest value changed from the base value/days to make the calculations
        list_of_params = [(0.2, 0.2, 14), (0.2, 0.15, 14), (0.1, 0.1, 14), (0.2, 0.1, 14), (0.1, 0.05, 14)]
        #for high in range(1, 20, 1):
         #   for low in range (1, 20, 1):
          #      for days in range (4, 60, 10):
           #         list_of_params.append((high * 0.01, low * 0.01, days))

        #Random Forest calssifier - test - for accuracy calculations
        clf_test_RF =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

        # Random Forest calssifier - actual - for final and exact culations
        clf_actual_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

        # K Neighbors Classifier - test - for accuracy calculations
        clf_test_KNN = KNeighborsClassifier(n_neighbors=3)
        # K Neighbors Classifier- actual - for final and exact culations
        clf_actual_KNN = KNeighborsClassifier(n_neighbors=3)
        # Support Vector Machine calssifier - test - for accuracy calculations
        clf_test_Svc = SVC()
        # Support Vector Machine calssifier - actual - for final and exact culations
        clf_actual_Svc = SVC()
        # Gradient Boosting calssifier - test - for accuracy calculations
        clf_test_Gradient = GradientBoostingClassifier(n_estimators=100)
        # Gradient Boosting calssifier - test - for accuracy calculations
        clf_actual_Gradient = GradientBoostingClassifier(n_estimators=100)

        # array that collects all strategies to be tested and value the results
        self._StrategyList = []

        # add all strategies to the array
        for item in list_of_params:
            str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
                                                    highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                    filePathIndexes="ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
                                                    highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                    filePathIndexes="ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                           filePathIndexes="ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
                                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                           filePathIndexes="ImportModule\\Indexes\\", clf_test = clf_test_RF, clf_actual = clf_actual_RF, clf_name="RF")
            self._StrategyList.append(str)

            # will be uncomment after I will understand how to make RECV to KNN & SVC classifiers
            # str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
            #                                         highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                         filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            # str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
            #                                         highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                         filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
            #                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
            #                                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test = clf_test_KNN, clf_actual = clf_actual_KNN, clf_name="KNN")
            # self._StrategyList.append(str)
            #
            # str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
            #                                         highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                         filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)
            # str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
            #                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
            #                                                highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                                filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)
            # str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
            #                                            highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
            #                                            filePathIndexes="..\\ImportModule\\Indexes\\", clf_test=clf_test_Svc,
            #                                         clf_actual=clf_actual_Svc, clf_name="SVC")
            # self._StrategyList.append(str)

            str = PercentForPeriodIndexCorrStrategy(name="PercentForPeriodIndexCorrStrategy",
                                                    highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                    filePathIndexes="ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)
            str = PercentForPeriodStrategy(name="PercentForPeriodStrategy",
                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                           filePathIndexes="ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexCorrRollingStrategy(name="PercentForPeriodIndexCorrRollingStrategy",
                                                           highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                           filePathIndexes="ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)
            str = PercentForPeriodIndexLogCorrStrategy(name="PercentForPeriodIndexLogCorrStrategy",
                                                       highestLimit=item[0], lowestLimit=item[1], hm_days=item[2],
                                                       filePathIndexes="ImportModule\\Indexes\\", clf_test=clf_test_Gradient,
                                                    clf_actual=clf_actual_Gradient, clf_name="Gradient")
            self._StrategyList.append(str)

    def DoStrategy(self, strategy, ticker, filename = "stock_ndx_processed\\processed_" , skipPredict = True):
        context = Context(strategy)
        acc, confusionmatrix, final = context.ProcessSpecificTicker(ticker , filename, skipPredict)
        return acc, confusionmatrix, final

    # find accuuracy & predict for specific ticker
    def PredictTicker(self, ticker, filename = "stock_ndx_processed\\processed_",skipPredict=False):

        try:
            print("Processing:{}".format(ticker))

            #labels for the Excel file
            labels = ['ticker' ,'name', 'clf','acc', 'BPSemi','BPAqurate', 'final','highest', 'lowest', 'Hm_days', '_-1 ', '_0 ', '_1 ', '_(0,1)',
                      '_(0,2) ', '_(1,0)', '_(1,2)', '_(2,0)', '_(2,1)']

            # interate all the strategies in the array
            for item in self._StrategyList:
                # get accuracy, cunfusion matrix and final decision from the specific strategy
                acc, confusionmatrix, final = self.DoStrategy(item, ticker,filename, skipPredict=False)
                # case error of case the ticker file is missing
                if (acc == None):
                    return None
                # BuyPercentAcc - buy signal that promise value will not fall under the lowest value
                BuyPercentAcc = 0
                denominator = confusionmatrix[0, 2] +  confusionmatrix[2, 0] + confusionmatrix[2, 2]
                if (denominator != 0):
                    BuyPercentAcc = confusionmatrix[2, 2] / (denominator)

                # BuyPercentAcc - accurate buy signal that predict value will go above the maximum
                # in other wors : not fall under the lowest value OR stay in between maximum & minimum
                BuyPercentAccAqurate = 0
                denominator = confusionmatrix[0, 2] + confusionmatrix[2, 0] + confusionmatrix[0, 1] + confusionmatrix[1, 0] + confusionmatrix[2, 2]
                if (denominator != 0):
                    BuyPercentAccAqurate = confusionmatrix[2, 2] / (denominator)

                # check the the classification managed to calculate for 3 diffrent labels and not 2(it happens)
                # the confusion matrix columns equals the number of labels
                x, y = confusionmatrix.shape
                if (x > 2):
                    c = np.array(
                        [[ticker, item.Name, item.Clf_Name, acc, BuyPercentAcc,BuyPercentAccAqurate, final[0], item.HighestLimit, item.LowestLimit, item.Hm_days, confusionmatrix[0, 0],
                          confusionmatrix[1, 1], confusionmatrix[2, 2], confusionmatrix[0, 1], confusionmatrix[0, 2],
                          confusionmatrix[1, 0],
                          confusionmatrix[1, 2], confusionmatrix[2,0], confusionmatrix[2, 1]]])

                    # add the row to the main df
                    if (len(self._df) > 0 ):
                        self._df = self._df.append(pd.DataFrame(c, columns=labels))
                    # create the main dataframe
                    else:
                        self._df = pd.DataFrame(c,  columns=labels)
                else:
                    print("Skipped...")

        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

    # interate the prediction over all the nasdaq tickers
    def PredictAll(self, ndxfilepath = '..\ImportModule\\ndx.csv', excelfilename = "strategyparams_processed\\complete.xlsx",
                    processfilename = "stock_ndx_processed\\processed_" , skipPredict=False, firstStockIndex = None, lastStockIndex = None, specificTicker = None):
        if (specificTicker == None):
            data = pd.read_csv(ndxfilepath)
            tickers = data['ticker']

            for ticker in tickers[firstStockIndex:lastStockIndex]:
                sys.stdout.write('.')
                self.PredictTicker(ticker, processfilename)

            # save all results to one excel file

            writer = pd.ExcelWriter(excelfilename)
            self._df.to_excel(writer)
            writer.save()
        else:
            self.PredictTicker(specificTicker, processfilename)
#pred = Predict()
#pred.PredictAll(skipPredict=False, firstStockIndex=5, lastStockIndex=8)
#print("END")


