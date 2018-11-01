import pandas as pd
import numpy as np

a = pd.read_csv('2.csv', low_memory=False)
b = pd.DataFrame(a)
c = b['time_interval'] = pd.to_datetime(b['Mention'], format="%Y/%m/%d", errors='coerce') - pd.to_datetime(
    b['Publication'], format="%Y/%m/%d", errors='coerce')


def f(i):
    return i / np.timedelta64(1, 'D')


result = c.apply(f)
print(result)
