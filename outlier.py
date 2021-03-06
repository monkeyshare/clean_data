from statsmodels import regression
import numpy as np
import pandas as pd
import math
from statsmodels import regression
#中位数去极致
def filter_extreme_MAD(series,n=3):
    series=pd.Series(series)
    median=series.quantile(0.5)
    new_median=((series-median).abs()).quantile(0.5)
    max_range=median+n*new_median
    min_range=median-n*new_median
    return np.clip(series,min_range,max_range)
def filter_extreme_3sigma(a,n=2,max_ignore=500):
    '''
    3sigma
    max_ignore:
    n:
    '''
    if max(a)<=max_ignore:
        n=n+2
    mean=np.mean(a)
    std=np.std(a)
    max_range=mean+n*std
    min_range=mean-n*std
    b=[]
    for i in a:
        if i>0 and i>min_range and i<max_range:
            b.append(i)
    result={}
    for position,i in enumerate(a):
        if i<max_range:
            result[position]=i
        elif i>=max_range:
            result[position]=round(np.mean(b),2)
    return list(result.values())
def filter_extreme_percentile(series,min=0.025,max=0.975):
    series=series.sort_values()
    q=series.quantile([min,max])
    return np.clip(series,q.iloc[0],q.iloc[1])
