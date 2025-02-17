from histocc.eval_metrics import EvalEngine
from histocc.prediction_assets import OccCANINE
import pandas as pd


df = pd.read_csv("Data/Training_data/DK_census_train.csv")
# df where hisco_2 is not " "
df2 = df[df.hisco_2 != " "]
df = df[0:10]
# Merge
df = pd.concat([df, df2[0:10]])
df = df.reset_index(drop = True)

mod = OccCANINE()

res = mod(df.occ1.tolist(), lang = "da")

eval_engine = EvalEngine(mod, ground_truth = df, predicitons = res, pred_col = "hisco_")
print(eval_engine.accuracy())
_ = 1

# acc = accuracy(ground_truth = df, predicitons = res, pred_col = "hisco_")


# print(res)


