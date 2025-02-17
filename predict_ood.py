
from histocc import OccCANINE
import pandas as pd
import os

mod = OccCANINE()

# list files
files = os.listdir('Data/OOD_data')

def int0(x):
    try:
        return int(x)
    except:
        return 0

for f in files:
    if f == 'Predictions':
        continue
    print(f'------> Predicting {f}')
    data_f = pd.read_csv(f'Data/OOD_data/{f}')
    res = mod(data_f.occ1)

    # If available compute (rough) accuracy
    if 'hisco_1' in data_f.columns:
        acc = 0
        for i in range(len(res)):
            correct_i = 0
            if int(res.hisco_1[i]) in [
                int0(data_f.hisco_1[i]), 
                int0(data_f.hisco_2[i]), 
                int0(data_f.hisco_3[i]), 
                int0(data_f.hisco_4[i]), 
                int0(data_f.hisco_5[i])
            ]:
                correct_i = 1
                acc += 1
            res.loc[i, 'correct'] = correct_i            
        acc = acc/len(res)
        print(f'==== Accuracy: {acc} ====')

    # Save predictions
    res.to_csv(f'Data/OOD_data/Predictions/{f[0:10]}_predictions.csv')
