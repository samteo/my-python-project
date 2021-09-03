# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:14:14 2021

@author: samte
"""

'Question 2'
def exist():
    if 'v' in globals():
        return print('v is defined')
    else:
        return print('v is not defined')        
exist()








import os
git ls-files "./*.py" | wc -l
git ls-files | xargs cat | sed '/^\s*$/d' | wc -l
git ls-files | xargs cat | grep -i -c def | wc -w
git diff --shortstat cba196ffb88501aa2ce087f75e5847c828f08087 3ecf5d05bddc8ff990681a536d7dc48c7cdfa94e
git clone git@github.com:samteo/my-python-project.git
def get_size():
    import os
    p=os.getcwd()+'\\my-python-project'
    total_size=0
    for f in os.listdir(p):
        if os.path.isfile(p+'\\'+f):
            total_size += os.path.getsize(p+'\\'+f)
        else:
            for s in os.listdir(p+'\\'+f):
                if os.path.isfile(p+'\\'+f+'\\'+s):
                    total_size += os.path.getsize(p+'\\'+f+'\\'+s)
    
    total_size_in_MB = total_size/1000000      
    return  total_size_in_MB









import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import scipy.optimize as sco
stock = ['AAPL','IBM','GOOG','BP','XOM','COST','GS']
weight = np.array([0.15,0.2,0.2,0.15,0.1,0.15,0.05])
data = pd.DataFrame()
for i in stock:
    msft = yf.Ticker(i)
    hist = msft.history(start="2016-01-01",end="2016-12-31")
    hist = hist[['Close']]
    returns = hist.pct_change()
    data[i] = returns
data=data.dropna()    

meanReturns = data.mean()
stdReturns = data.std()
covMatrix = data.cov()
data['portfolio'] = data.dot(weight)

def historicalVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)

    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")

historicalVaR(returns=data['portfolio'], alpha=5)

def historicalCVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the CVaR for dataframe / series
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()

    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")

def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(Time)
    return returns, std

pRet,pStd = portfolioPerformance(weight, meanReturns, covMatrix, 252)



def var_parametric(portofolioReturns, portfolioStd, alpha=5):
    VaR = norm.ppf(1-alpha/100)*portfolioStd - portofolioReturns
    return VaR

def cvar_parametric(portofolioReturns, portfolioStd, alpha=5):
    CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*portfolioStd - portofolioReturns
    return CVaR

var_p = var_parametric(pRet,pStd, alpha=5)
cvar_p = cvar_parametric(pRet,pStd, alpha=5)






def calc_portfolio_std(weight, meanReturns, covMatrix):
    portfolio_std = np.sqrt(np.dot(weight.T, np.dot(covMatrix, weight))) * np.sqrt(252)
    return portfolio_std
def min_variance(meanReturns, covMatrix):
    num_assets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_std, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
min_port_variance = min_variance(meanReturns, covMatrix)

pd.DataFrame([round(x,2) for x in min_port_variance['x']],index=tickers).T


data=data.drop(columns=['portfolio'])
for i in range(1,13):
    eachMonth = '2016-'+'%02d'%i
    eachMonthData = data.loc[eachMonth]
    meanReturnMonthly = eachMonthData.mean()
    stdReturnMonthly= eachMonthData.std()
    print(meanReturnMonthly,stdReturnMonthly)
    






'Question 6'
import re
text = '2009/09/04 hahaha 12/24/2009 hahaha 04/12/2009 hahah 19 Jan 1992 19 Sept 1992'
date_reg_exp1 = re.compile('\d{4}[/]\d{2}[/]\d{2}')
date_reg_exp2 = re.compile('\d{2}[/]\d{2}[/]\d{4}')
date_reg_exp3 = re.compile('\d{2}\s[A-Z]+[a-z]{2,3}\s\d{4}')
regex_list = [date_reg_exp1,date_reg_exp2,date_reg_exp3]
date=[]

for r in regex_list:
    matches_list=r.findall(text)
    for m in matches_list:
        date.append(m)

print(len(date))













