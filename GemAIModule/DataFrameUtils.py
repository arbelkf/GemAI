# Auther : Kfir Arbel
# date : 6.8.2018
# DataFrameUtils Class

import pandas as pd
import os
import definitions

class DataFrameUtils():
    # get all the data about a ticker from a saved file -
    # return also all features that are in the featurelist param
    def GetFeaturesFromCSV(self, fileName, featureList):
        try:
            absoluteFileName = os.path.join(definitions.ROOT_DIR, fileName)
            if (os.path.isfile(absoluteFileName)):
                df = pd.read_csv(absoluteFileName )
                df2 = pd.DataFrame()
                df.set_index('Date', inplace=True)
                #df2 = pd.concat([df2, df[['pct_1d']]], axis=1)
                for index in featureList:
                    df2 = pd.concat([df2, df[[index]]], axis=1)


                df2.dropna(inplace=True)
                return df2
            else:
                print("file {} missing:".format(absoluteFileName))
                return None
        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])

    # add all the data about a ticker from a saved file to an existing dataframe -
    # return also all features that are in the featurelist param
    def GetFeaturesFromCSVToExistingDF(self, fileName, featureList):
        try:
            dftemp = pd.DataFrame()
            absoluteFileName = os.path.join(definitions.ROOT_DIR, fileName)
            if (os.path.isfile(absoluteFileName)):
                df = pd.read_csv(absoluteFileName)

                if (len(df) > 0):
                    df.set_index('Date', inplace=True)

                    for index in featureList:
                        dftemp = pd.concat([dftemp, df[[index]]], axis=1)
                    dftemp.dropna(inplace=True)

            else:
                print("file {} missing:".format(absoluteFileName))

            return dftemp
        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])
        return dftemp

    # get only one column from the dataframne
    def GetOnlyOneColumn(self, df, columnName):
        try:
            df2 = pd.DataFrame()
            #df.set_index('Date', inplace=True)
            df2 = pd.concat([df2, df[[columnName]]], axis=1)

            df2.dropna(inplace=True)
            return df2

        except ValueError  as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            print("args:", inst.args[0])
