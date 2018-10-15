import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
df=sm.datasets.get_rdataset("Guerry", "HistData").data
df=df.dropna()
vars=['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df=df[vars]
y,X=dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
models0=sm.OLS(y,X)
res=models0.fit()
print(res.summary())
for p,k in zip(list(res.pvalues),list(res.params)):
    print(k,p)
res.predict([1,1,0,0,0,30,73])
res.predict(X[:5])
