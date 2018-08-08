# Auther : Kfir Arbel
# date : 6.8.2018
# Main Class


from GemAIModule.Predict import Predict
import sys

str = sys.stdin.readline()
while(str != "END\n"):
    strparam = str.split()
    if (strparam[0] == "pred"):
        pred = Predict()
        pred.PredictAll(ndxfilepath = 'ImportModule\\ndx.csv',processfilename="GemAIModule\\stock_ndx_processed\\processed_",
                        skipPredict=False, firstStockIndex=7, lastStockIndex=7, specificTicker=strparam[1])
        print("END")
    str = sys.stdin.readline()
