import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NDXfile = os.path.join(ROOT_DIR, 'ImportModule\\ndx.csv')
PROCESSFilePath = os.path.join(ROOT_DIR, "GemAIModule\\stock_ndx_processed\\processed_")
EXCELFile = os.path.join(ROOT_DIR, "GemAIModule\\strategyparams_processed\\complete.xlsx")