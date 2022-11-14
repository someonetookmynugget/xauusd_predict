

# 共通ライブラリをインストール
# 数値計算ライブラリ
import numpy as np 
import pandas as pd

# グラフ化
import matplotlib.pyplot as plt
import seaborn as sns # pip install seaborn

# 設定
import warnings
warnings.filterwarnings("ignore")


df_pass = pd.read_csv("xauusd.csv", encoding="shift_jis")
import datetime as dt
for i, d in enumerate(df_pass["Date"]):
    if len(df_pass["Date"][i].split("/")[1]) != 2 and len(df_pass["Date"][i].split("/")[2]) != 2:
        df_pass["Date"][i] = df_pass["Date"][i].split("/")[0] + "/0" + df_pass["Date"][i].split("/")[1] +"/0"+df_pass["Date"][i].split("/")[2]
    elif len(df_pass["Date"][i].split("/")[1]) != 2:
        df_pass["Date"][i] = df_pass["Date"][i].split("/")[0] + "/0" + df_pass["Date"][i].split("/")[1] +"/"+df_pass["Date"][i].split("/")[2]
    elif len(df_pass["Date"][i].split("/")[2]) != 2:
        df_pass["Date"][i] = df_pass["Date"][i].split("/")[0] + "/" + df_pass["Date"][i].split("/")[1] +"/0"+df_pass["Date"][i].split("/")[2]



df_pass["Date"] = pd.to_datetime(df_pass["Date"],format="%Y/%m/%d")
df_pass=df_pass.set_index("Date")
array_pass = df_pass["Close"].values
# インプットデータを可視化
# plt.figure()
# plt.plot(array_pass,label="past")
# plt.grid()
# plt.legend()
# plt.show()

import statsmodels.api as sm

ORDER_P = 5
ORDER_D = 1
ORDER_Q = 3

model = sm.tsa.statespace.SARIMAX(array_pass,order=(ORDER_P,ORDER_D,ORDER_Q),seasonal_order=(0,0,1,12),trend='c',enforce_invertibility=False)
result = model.fit(trend='c' )
print(result.summary())

pred = result.predict(3275, 4000)
print("prediction size =",pred.shape)
print("prediction values =",pred)

# 予測データも可視化
plt.figure()
plt.plot(array_pass, label="past")
plt.plot(np.arange(array_pass.all()),pred, label="prediction") #予測値リストのサイズに合うようにrangeの区間を設定。
plt.grid()
plt.legend()
plt.show()