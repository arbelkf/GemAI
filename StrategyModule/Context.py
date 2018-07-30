


class Context:
    """
    Define the interface of interest to clients.
    Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def ProcessTicker(self, filename , ticker, skipPredict = False):
        acc, confusionmatrix, final = self._strategy.ProcessTicker(filename, ticker, skipPredict)
        return acc, confusionmatrix, final
        #print(self._strategy.Name)

    def ProcessComposedTicker(self, filename , ticker, skipPredict = False):
        acc, confusionmatrix, final = self._strategy.ProcessComposedTicker(filename, ticker, skipPredict)
        return acc, confusionmatrix, final