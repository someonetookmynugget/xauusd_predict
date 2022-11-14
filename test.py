import datetime
import pandas as pd

today = "2022/10/31"
today = pd.to_datetime(today, format="%Y/%m/%d")
for i in range(1, 366):
    day_after_tomorrow = datetime.timedelta(days=i)
    print(today+day_after_tomorrow)