# https://qiita.com/nobu6787/items/a2e927e3ccbe04d9700f#1%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm # pip install statsmodels
from statsmodels.tsa.arima_model import ARIMA
# %matplotlib inline
import matplotlib.pyplot as plt


stock_xauusd = pd.read_csv("xauusd.csv", encoding="shift_jis")
#日付整備
import datetime as dt

#日付をdatetime型に変換
for i, d in enumerate(stock_xauusd["Date"]):
    if len(stock_xauusd["Date"][i].split("/")[1]) != 2 and len(stock_xauusd["Date"][i].split("/")[2]) != 2:
        stock_xauusd["Date"][i] = stock_xauusd["Date"][i].split("/")[0] + "/0" + stock_xauusd["Date"][i].split("/")[1] +"/0"+stock_xauusd["Date"][i].split("/")[2]
    elif len(stock_xauusd["Date"][i].split("/")[1]) != 2:
        stock_xauusd["Date"][i] = stock_xauusd["Date"][i].split("/")[0] + "/0" + stock_xauusd["Date"][i].split("/")[1] +"/"+stock_xauusd["Date"][i].split("/")[2]
    elif len(stock_xauusd["Date"][i].split("/")[2]) != 2:
        stock_xauusd["Date"][i] = stock_xauusd["Date"][i].split("/")[0] + "/" + stock_xauusd["Date"][i].split("/")[1] +"/0"+stock_xauusd["Date"][i].split("/")[2]



stock_xauusd["Date"] = pd.to_datetime(stock_xauusd["Date"],format="%Y/%m/%d")

# print(stock_xauusd["Date"])
#datetime型にした日付をインデックス化
stock_xauusd=stock_xauusd.set_index("Date")

for i in range(1, 1096):
    day_after_tomorrow = dt.timedelta(days=i)
    stock_xauusd.loc[stock_xauusd.last_valid_index()+day_after_tomorrow] = None


#降順に並んでいるデータを昇順に変換
stock_xauusd = stock_xauusd.sort_index()

#終値の取得とDataFrame型への変換
    #データには「始値」「高値」「安値」「終値」の４種類があるため、ここでは終値ベースで予測
stock_xauusd_end = stock_xauusd.loc[:,"Close"]
stock_xauusd_end = pd.DataFrame(stock_xauusd_end)

#データ数値の整備
    #float型に変換
stock_xauusd_end["Close"] = stock_xauusd_end["Close"].astype(float)

#パラメーターの最適化
        #各パラメーターがそれぞれ、0〜2の場合についてのSARIMAモデルのBICを計算し、最もBICが小さくなった場合を表示。
        #BICの場合は この値が低ければ低いほどパラメーターの値は適切であるため
# def selectparameter(DATA, s):
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
#     parameters = []
#     BICs = np.array([])
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(DATA,
#                                                 order=param,
#                                                 seasonla_order=param_seasonal)

#                 results = mod.fit()
#                 parameters.append([param, param_seasonal, results.bic])
#                 BICs = np.append(BICs, results.bic)
#             except:
#                 print("error")
#                 continue
        
#     return parameters[np.argmin(BICs)]
#年毎の周期を確認したので、周期は12とする
# best_params = selectparameter(stock_xauusd_end, 12)#prob 365?



ORDER_P = 5
ORDER_D = 1
ORDER_Q = 3

# SARIMAX
model = sm.tsa.statespace.SARIMAX(stock_xauusd_end,order=(ORDER_P,ORDER_D,ORDER_Q),seasonal_order=(0,0,1,12),trend='c',enforce_invertibility=False)
result = model.fit(trend='c' )

# SARIMA_stock_xauusd = sm.tsa.statespace.SARIMAX(stock_xauusd_end, order=best_params[0],seasonal_order=best_params[1],
#                                                 enforce_stationarity=False, enforce_invertibility=False).fit()

pred = result.predict("2022-09-01", "2025-10-31")

plt.plot(stock_xauusd_end,label="past") 
plt.plot(pred, "r",label="prediction") 
plt.grid()
plt.legend()
plt.show()

print(pred)
