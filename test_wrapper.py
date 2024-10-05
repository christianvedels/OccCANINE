# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:49:33 2024

@author: christian-vs
"""

from histocc.prediction_assets import OccCANINE
import pandas as pd

if __name__ == "__main__":
    # mod = OccCANINE(hf = True, batch_size = 256, verbose = True)
    # mod = OccCANINE(name = "Data/models/mixer-s2s=10-s=1605000.bin", hf = False, batch_size = 16, verbose = True)
    mod = OccCANINE(name = "Data/models/baseline-s=1600000.bin", hf = False, batch_size=4)
    # mod = OccCANINE(name = "Data/models/flat_occCANINE.bin", hf = False, verbose = True, batch_size = 2)
    
    # print(mod.predict(["he is a farmer", "he is a fisherman"], behavior="fast", what = "probs"))
    # print(mod.predict_old(["he is a farmer"]))
    
    toydata = pd.read_csv("histocc/Data/TOYDATA.csv")
    # df = pd.read_csv("d:/dropbox/research_projects/conflict, religion, and development/wp2 - danish religious conflict in the us/wp2_replication_package/data/non_redistributable/occstrings.csv")
    # res = mod.predict(df.occ1.tolist()[0:100], lang = "en")
    res = mod.predict(toydata[0:8].occ1, lang = "en", what = "pred", prediction_type = "full")
    # res = mod.prkedict(toydata[0:8].occ1, lang = "en", behavior = 'fast')
    # res = mod.predict(["hod carriers", "han staar i tjeneste"], lang = "en")
    print(res)
    # print(mod.predict_old(toydata.occ1.tolist()[0:1000]))
    
    # print(mod.predict_old(["he is a farmer"]))
