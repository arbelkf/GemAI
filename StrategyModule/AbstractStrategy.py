# Auther : Kfir Arbel
# Abstract Stratgegy -
# Base class to all strategies

import abc
import pandas as pd
import numpy as np

from GemAIModule.DataFrameUtils import DataFrameUtils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)




class AbstractStrategy(object , metaclass=abc.ABCMeta):
    """
    Declare an interface for a type of product object.
    """

    def __init__(self,  name = "ProcessedStocks",isIndicatorLongList = False, filePathProcessedStocks = "ProcessedStocks", filePathIndexes = "Indexes",
                 highestLimit =  0.02, lowestLimit = 0.02, hm_days = 7, isDaily = True, spikeLowest = 0.05, spikeHighest = 0.05,
                 clf_name = "RF",
                  clf_test =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                      max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
                      verbose=0, warm_start=False),
                 clf_actual=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                                            min_samples_leaf=1,
                                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                                            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
                                            verbose=0, warm_start=False)
                 ):
        self._dfExisting = pd.DataFrame()
        self._isDaily = isDaily
        self._name = name
        self._hm_days = hm_days
        self._highestLimit =  highestLimit
        self._lowestLimit = lowestLimit
        self._spikeLowest = spikeLowest
        self._spikeHighest = spikeHighest
        self._clf_test = clf_test
        self._clf_actual = clf_actual
        self._clf_name = clf_name
        if (isDaily == True):
            self._spFileLocation = 'sp500_joined_closes.csv'
        else:
            self._spFileLocation = 'sp500_weekly_joined_closes.csv'
        self._indexesList = ['^DJI', '^GDAXI', '^HSI', '^FCHI', '^GSPC', '^IXIC', '^N225', '^RUT', '^TYX']
        self._filePathIndexes = filePathIndexes
        if (isDaily == True):
            self._filePathProcessedStocks = filePathProcessedStocks
        else:
            self._filePathProcessedStocks = "Weekly_" + filePathProcessedStocks
        self._filePathStocks = "stock_dfs"
        if (isIndicatorLongList == False):
            self._indicators = ['atr', 'boll', 'boll_ub','boll_lb','open_2_sma','cr-ma3', 'volume_-3~1_min',
                                'vr_6_sma','cr-ma2','trix_9_sma','volume_-3,2,-1_max','vr',
                                'macds','adxr','cr-ma1','dma']
            #self._indicators = ['atr', 'boll']

        else:
            self._indicators = ['volume_delta', 'open_-2_r', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'volume_-3,2,-1_max', 'volume_-3~1_min',
                      'kdjk', 'kdjd', 'kdjj', 'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb',
                      'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 'wr_10', 'wr_6', 'cci', 'cci_20', 'tr',
                      'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'adxr', 'trix', 'trix_9_sma', 'vr', 'vr_6_sma']

        self._baseFileFolder = os.path.dirname(__file__)

        self._featureList = ['adj close', 'atr', 'boll_ub', 'cr-ma3', 'volume_-3~1_min',
                             'vr_6_sma', 'cr-ma2', 'trix_9_sma', 'volume_-3,2,-1_max', 'vr',
                             'macds', 'adxr', 'cr-ma1', 'dma']
        print("AbstractStrategy")

    @property
    def IsDaily(self):
        return self._isDaily

    @property
    def Hm_days(self):
        return self._hm_days

    @property
    def Name(self):
        return self._name

    @property
    def HighestLimit(self):
        return self._highestLimit

    @property
    def SpikeLowest(self):
        return self._spikeLowest

    @property
    def SpikeHighest(self):
        return self._spikeHighest

    @property
    def LowestLimit(self):
        return self._lowestLimit


    @property
    def IndexesList(self):
        return self._indexesList


    @property
    def SPFileLocation(self):
        abs_file_path = os.path.join(self._baseFileFolder, "../" + self._spFileLocation)
        return abs_file_path

    @property
    def Indicators(self):
        return self._indicators

    @property
    def FilePathProcessedStocks(self):
        abs_file_path = os.path.join(self._baseFileFolder, "../" + self._filePathProcessedStocks)
        return abs_file_path

    @property
    def FilePathStocks(self):
        abs_file_path = os.path.join(self._baseFileFolder, "../" + self._filePathStocks)
        return abs_file_path


    @property
    def Clf_Name(self):
        return self._clf_name

    @property
    def BaseFileFolder(self):
        return self._baseFileFolder

    @abc.abstractmethod
    def ExtractLabels(self, dfdata):
        raise NotImplementedError("Please Implement this method")

    ''''@abc.abstractmethod
    def ProcessData(self, ticker):
        raise NotImplementedError("Please Implement this method")'''

    @abc.abstractmethod
    def process_data_for_labels(self, dfdata):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def buy_sell_hold(self,*args):
        raise NotImplementedError("Please Implement this method")

    # add indexes to the dataframe
    # ticker = name of the index
    # df - the datafarem to add the index to
    def AddIndex(self, ticker, df):
        df2 = pd.read_csv(self._filePathIndexes + '\{}.csv'.format(ticker))
        df2.set_index('Date', inplace=True)
        df2.rename(columns={'Adj Close': ticker}, inplace=True)
        df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        df = df.join(df2, how='inner')

        df[ticker] = df[ticker].pct_change()
        df = df.replace([np.inf, -np.inf], 0)
        df.fillna(0, inplace=True)
        return df

    # clean the dataframe from np.infinity, -np.infinity and drop all NAN
    def CleanDF(self, dfdata):
        dfdata.fillna(0, inplace=True)
        dfdata = dfdata.replace([np.inf, -np.inf], np.nan)
        dfdata.dropna(how='all', inplace=True)
        return dfdata



    def SetActualClf(self):
        raise NotImplementedError("Please Implement this method")

    def concatdf(self, dftemp, dfExisting):
        if (len(dftemp) > 0):
            #################
            rows, columns = dftemp.shape
            numoffeature = columns - 1

            dftemp = dftemp[(dftemp.T != 0).any()]

            # extract labels from the data
            # the function returns y column, datafram and prediction
            y, dftemp, pred = self.ExtractLabels(dftemp)

            if (len(dfExisting) < 1):
                dfExisting = dftemp
            else:
                dfExisting = dfExisting.append(dftemp)
        return dfExisting
    # collect all the data from all tickers to one dataframe and pass to the ProcessTicker
    def ProcessComposedTicker(self, filename , ticker, skipPredict = False):
        print("Processing using {} clf {} high {} low {} hm {}".format(self._name, self.Clf_Name, self._highestLimit,
                                                                       self._lowestLimit, self._hm_days))
        dfUtils = DataFrameUtils()
        filepath = '..\ImportModule\\ndx.csv'
        dfUtils = DataFrameUtils()
        data = pd.read_csv(filepath)
        ndxtickers = data['ticker']
        dfExisting = pd.DataFrame()
        # collect all ticker to one dataframe except the ticker that will be sasved for the last records
        for ndxticker in ndxtickers[5:15]:
            if (ndxticker != ticker):
                sys.stdout.write('.')
                dftemp = dfUtils.GetFeaturesFromCSVToExistingDF(filename + ndxticker + ".csv", self._featureList)
                dfExisting = self.concatdf(dftemp, dfExisting)

        # adding the ticker to the last of the dataframe - that way - the prediction will be created to the latest data from the relevant ticker
        dftemp = dfUtils.GetFeaturesFromCSVToExistingDF(filename + ticker + ".csv", self._featureList)
        dfExisting = self.concatdf(dftemp, dfExisting)

        if (len(dfExisting ) < 1):
            return None, None, None

        y = dfExisting['target'].values
        pred = dfExisting.iloc[-1:]

        # save all results to one excel file
        filename = "strategyparams_processed\\temp.xlsx"
        writer = pd.ExcelWriter(filename)
        dfExisting.to_excel(writer, '{}'.format(ticker))
        writer.save()



        acc, confusionmatrix, final = self.ProcessTicker(dfExisting,y, pred,skipPredict)
        return acc, confusionmatrix, final

    # collect data for specific ticker and pass to the ProcessTicker
    def ProcessSpecificTicker(self, filename , ticker, skipPredict = False):
        print("Processing using {} clf {} high {} low {} hm {}".format(self._name, self.Clf_Name, self._highestLimit, self._lowestLimit, self._hm_days))
        dfUtils = DataFrameUtils()
        # get the data for the ticker from the data scrapped before
        df = dfUtils.GetFeaturesFromCSV(filename + ticker + ".csv", self._featureList)
        acc, confusionmatrix, final = self.ProcessTicker(df, skipPredict)
        return acc, confusionmatrix, final

    # process ticker
    # extract labels
    # do strategy
    # calculate accuracy and make prediction
    def ProcessTicker(self, df, y, pred, skipPredict = False):
        # in case the data for the ticker is missing
        # if (df is None):
        #     return None, None, None
        # rows, columns = df.shape
        # numoffeature = columns - 1
        #
        #
        # df = df[(df.T != 0).any()]
        #
        # # extract labels from the data
        # # the function returns y column, datafram and prediction
        # y, df , pred = self.ExtractLabels(df)

        X = df.ix[1:,:-1].values

        # the only prediction that count is the last one
        pred = pred.ix[:, :-1].values

        y = np.delete(y, (0), axis=0)
        # normalize the data for X
        X = preprocessing.StandardScaler().fit_transform(X)

        # spilt the data for the accuracty calculations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # get the test classifier
        clf_test = self._clf_test

        # fir the test classifier
        clf_test.fit(X_train, y_train)
        # predict from the X_test
        y_pred = clf_test.predict(X_test)
        # compare y_test o the x_test following the classifier that was trained before on the train data
        acc = clf_test.score(X_test, y_test)
        print("accuracy : {}".format(acc))
        # calculate the confusion matrix
        confusionmatrix = confusion_matrix(y_test,y_pred, labels=[-1, 0, 1] )

        final = [0]
        if (skipPredict == True):
            #print(confusionmatrix)
            return acc, confusionmatrix, final

        # get the actual classifier
        clf = self._clf_actual
        # use the actual classifier and recursive feature elimination to select the best number of features.
        m = RFECV(clf, scoring='accuracy')
        # fit the classifier
        m.fit(X, y)
        # predict the final recommedation/decision
        final = m.predict(pred)

        print("Final:{}".format(final))
        return acc, confusionmatrix, final