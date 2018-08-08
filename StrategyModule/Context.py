# Auther : Kfir Arbel
# date : 6.8.2018
# Context Class


class Context:
    """
    Define the interface of interest to clients.
    Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def ProcessSpecificTicker(self, ticker, filename = "stock_ndx_processed\\processed_" , skipPredict = False):
        acc, confusionmatrix, final = self._strategy.ProcessSpecificTicker(ticker,filename, skipPredict)
        return acc, confusionmatrix, final
        #print(self._strategy.Name)

    def ProcessComposedTicker(self, ticker, filename = "stock_ndx_processed\\processed_" ,skipPredict = False):
        acc, confusionmatrix, final = self._strategy.ProcessComposedTicker(ticker, filename, skipPredict)
        return acc, confusionmatrix, final