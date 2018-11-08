# https://www.jianshu.com/p/3afbb01c9ed6
# https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/
# 简单平均数

# 移动平均数
# 指数平滑法
# holt线性趋势法
# holt冬季季节法
# ARIMA
import  pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from pyramid.arima import auto_arima

t=pd.read_excel('all_1.xlsx',encoding='gbk')
g=t.groupby('时间').agg({'人数':np.mean})
train=g[:300]
test=g[300:]
y_hat=test.copy()
y_hat['naive']=list(train['人数'])[-1]
y_hat['avg_forcast']=np.mean(train['人数'])
y_hat['moving_avg_forcast']=train['人数'].rolling(6).mean().iloc[-1]
fit2 = SimpleExpSmoothing(np.asarray(train['人数'])).fit(smoothing_level=0.6,optimized=False)
y_hat['SES']=fit2.forecast(test.shape[0])

fit1 = Holt(np.asarray(train['人数'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat['Holt_linear'] = fit1.forecast(test.shape[0])

fit3 = ExponentialSmoothing(np.asarray(train['人数']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat['Holt_Winter'] = fit3.forecast(test.shape[0])

autoarimamodel = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
autoarimamodel.fit(train)
y_hat['autoarima'] = autoarimamodel.predict(n_periods=test.shape[0])


fit1 = sm.tsa.statespace.SARIMAX(train['人数'], order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat['SARIMA'] = fit1.predict(start=list(test.index)[0], end=list(test.index)[-1], dynamic=True)
'''
# 分离季节性、趋势性
sm.tsa.seasonal_decompose(train['人数']).plot()
result=sm.tsa.stattools.adfuller(train['人数'])
plt.show()
'''
from pyramid.arima import auto_arima





plt.figure(figsize=(12,8))
plt.plot(train.index, train['人数'], label='Train')
plt.plot(test.index,test['人数'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.plot(y_hat.index,y_hat['avg_forcast'], label='avg_forcast')
plt.plot(y_hat.index,y_hat['moving_avg_forcast'], label='moving_avg_forcast')
plt.plot(y_hat.index,y_hat['SES'], label='SES')
plt.plot(y_hat.index,y_hat['Holt_linear'], label='Holt_linear')
plt.plot(y_hat.index,y_hat['Holt_Winter'], label='Holt_Winter')
plt.plot(y_hat.index,y_hat['SARIMA'], label='SARIMA')
plt.plot(y_hat.index,y_hat['autoarima'], label='autoarima')

plt.legend(loc='best')
plt.title("Forecast")
plt.show()

rms=sqrt(mean_squared_error(y_hat['人数'],y_hat['Holt_linear']))
print(rms)
