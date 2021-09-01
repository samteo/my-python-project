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




'Question3'
import pandas as pd
import numpy as np
import yfinance as yf
stock = ['AAPL','IBM','GOOG','BP','XOM','COST','GS']
weight = [0.15,0.2,0.2,0.15,0.1,0.15,0.05]
data = pd.DataFrame()
for i in stock:
    msft = yf.Ticker(i)
    hist = msft.history(start="2016-01-01",end="2016-12-31")
    hist = hist[['Close']]
    returns = hist.pct_change()
    data[i] = returns
data=data.dropna()    
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













