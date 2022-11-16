import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime

date_st = datetime(2010,3, 1)

date_dn = datetime(2022,11,4)
df = pd.read_csv("xauusd.csv")
df.head()