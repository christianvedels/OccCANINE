# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:49:33 2024

@author: christian-vs
"""

from histocc.prediction_assets import OccCANINE
import pandas as pd

if __name__ == "__main__":
    mod = OccCANINE(name = "Data/models/mixer-s2s=10-s=1605000.bin", hf = False, batch_size = 256, verbose = True)
    # mod = OccCANINE(name = "Data/models/baseline-s=1600000.bin", hf = False)
    # mod = OccCANINE(name = "Data/models/flat_occCANINE.bin", hf = False, verbose = True, batch_size = 128)
    
    # print(mod.predict(["he is a farmer", "he is a fisherman"], behavior="fast", what = "probs"))
    # print(mod.predict_old(["he is a farmer"]))
    
    
    toydata = pd.read_csv("histocc/Data/TOYDATA.csv")
    res = mod.predict(toydata.occ1.tolist()[0:100], lang = "en", behavior="good", what = "pred")
    print(res)
    # print(mod.predict_old(toydata.occ1.tolist()[0:1000]))
    
    # print(mod.predict_old(["he is a farmer"]))
