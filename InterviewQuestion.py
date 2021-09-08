# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:14:14 2021

@author: samte
"""
'''Question 1'''
#a
import pandas as pd
import numpy as np
name=['John','Mike','Sally','Jane','Joe','Dan','Phil']
salary=[300,200,550,500,600,600,550]
mi=[3,3,4,7,7,3,'NULL']
t = pd.DataFrame(data={'Name':name,'Salary':salary,'Manager_id':mi})
t.index = np.arange(1, len(t)+1)
manager = t['Manager_id'].unique()
for i in manager:
    if str(i).isdigit():
        employee= t[t['Manager_id']==i]['Salary']
        salary_higher = (employee[employee> t.loc[i,'Salary']]).index.item()
        print('Employee {} salary is higher than his manager'.format(t.loc[salary_higher,'Name']))
'''
Output:        

Employee Dan salary is higher than his manager
Employee Sally salary is higher than his manager
Employee Joe salary is higher than his manager
'''

#b
manager = set(manager)
not_manager=set(t.index).difference(manager)
avg_salary = t[t.index.isin(not_manager)]['Salary'].mean()
print(avg_salary)
'''
Output:        
425.0
'''

'''----------------------------------------------------------------------------------------'''

'''Question 2'''
def exist():
    if 'v' in globals():
        return True
    else:
        return False 
exist()

'''
Output:        
False
'''

'''----------------------------------------------------------------------------------------'''

'''Question 3'''
def pascals_triangle(n_layers):
    result=[]
    for n in range(n_layers):
        row=[1]
        if result:
            last_row=result[-1]
            row.extend([sum(p) for p in zip(last_row,last_row[1:])])
            row.append(1)
        result.append(row)
        # print(str(row))
        print(' '.join(map(str, row)))    
pascals_triangle(6)
'''
Output:
1
1 1
1 2 1
1 3 3 1
1 4 6 4 1
1 5 10 10 5 1
'''

'''----------------------------------------------------------------------------------------'''

'''Question 4'''
#Getting the data
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import scipy.optimize as sco
import seaborn as sns
stock = ['AAPL','IBM','GOOG','BP','XOM','COST','GS']
weight = np.array([0.15,0.2,0.2,0.15,0.1,0.15,0.05])
data = pd.DataFrame()
for i in stock:
    stockData = yf.Ticker(i)
    hist = stockData.history(start="2016-01-01",end="2016-12-31")
    hist = hist[['Close']]
    returns = hist.pct_change()
    data[i] = returns
data=data.dropna(axis=0,how='all')    

meanReturns = data.mean()
stdReturns = data.std()
covMatrix = data.cov()
portfolio = data.dot(weight)


#a
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


        
historicalVaR(returns=portfolio , alpha=5)        
historicalCVaR(returns=portfolio , alpha=5)


#b
def portfolioPerformance(weight, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weight)*Time
    std = np.sqrt( np.dot(weight.T, np.dot(covMatrix, weight)) ) * np.sqrt(Time)
    return returns, std

pRet,pStd = portfolioPerformance(weight, meanReturns, covMatrix, 252)



def var_parametric(portofolioReturns, portfolioStd, alpha=5):
    VaR =  portofolioReturns - norm.ppf(1-alpha/100)*portfolioStd
    return VaR

def cvar_parametric(portofolioReturns, portfolioStd, alpha=5):
    CVaR =  portofolioReturns - (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*portfolioStd
    return CVaR
sns.displot(portfolio)
var_p = var_parametric(pRet,pStd, alpha=5)
cvar_p = cvar_parametric(pRet,pStd, alpha=5)



#c

import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt


df_data = data.DataReader(['AAPL','IBM','GOOG','BP','XOM','COST','GS'], 'yahoo', start='2016/01/01', end='2016/12/31')
df_data  = df_data['Adj Close']
df_data.head()

for m in range(1,13): #iterate each months
    eachMonth = '2016-'+'%02d'%m
    df = df_data.loc[eachMonth]
    df_pct_change_all=df.pct_change().apply(lambda x: np.log(1+x)).dropna()
    
    df_mean =  df_pct_change_all.mean()
    df_positive=df_mean[df_mean>0]
    df_negative=df_mean[df_mean<0]
    group = [df_positive.index.tolist(),df_negative.index.tolist()]
    for g in group:

            
        df_pct_change=df_pct_change_all[g].dropna()
        if group.index(g)==1:
            # group to short
            df_pct_change=df_pct_change*-1
                    
        cov_matrix = df_pct_change.cov()
        cov_matrix
    
        corr_matrix = df_pct_change.corr()
        corr_matrix
        weights = np.random.random(len(g))
        weights/=np.sum(weights)
        
        w = {g[i]:weights[i] for i,o in enumerate(g)}
        
        # w = {'AAPL': weights[0], 'IBM': weights[1], 'GOOG': weights[2], 'BP': weights[3],'XOM':weights[4],'COST':weights[5],'GS':weights[6]}
        port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
        port_var
    
        ind_er = df_pct_change.mean()
        ind_er
        
        w = list(weights)
        port_er = (w*ind_er).sum()
        port_er
        
        month_sd = df_pct_change.std().apply(lambda x: x*np.sqrt(19))
        month_sd
        
        assets = pd.concat([ind_er, month_sd], axis=1) # Creating a table for visualising returns and volatility of assets
        assets.columns = ['Returns', 'Volatility']
        assets
        
        p_ret = [] # Define an empty array for portfolio returns
        p_vol = [] # Define an empty array for portfolio volatility
        p_weights = [] # Define an empty array for asset weights
        
        num_assets = len(g)
        num_portfolios = 100
        
        for portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights = weights/np.sum(weights)
            p_weights.append(weights)
            returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                              # weights 
            p_ret.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
            sd = np.sqrt(var) # Daily standard deviation
            month_sd = sd*np.sqrt(19) # Annual standard deviation = volatility
            p_vol.append(month_sd)
        data = {'Returns':p_ret, 'Volatility':p_vol}
        
        for counter, symbol in enumerate(g):
            #print(counter, symbol)
            data[symbol+' weight'] = [w[counter] for w in p_weights]
        
        portfolios  = pd.DataFrame(data)
        portfolios.head() # Dataframe of the 10000 portfolios created
        # Plot efficient frontier
        # portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])
        min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
        # idxmin() gives us the minimum value in the column specified. 
        if group.index(g)==0:                             
            print('Month {} (Long)'.format(m))
        else:
            print('Month {} (Short)'.format(m))
        print(min_vol_port)
        # plt.subplots(figsize=[10,10])
        # plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
        # plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
        
'''
The idea come from https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/,
but I revamped it to a version that can be used in LONG and SHORT scenarios. 
With my limitated knowledge in financal backgroud, I am not sure if it can be used in practical but it is what i have from my research.
Assumption: You have to defined how much of portion of money you want to perform LONG or SHORT in that month.
            The sum of the weights in LONG or SHORT which are showing below is 1 respectively.
Output :
    
Month 1 (Long)
Returns        0.000400
Volatility     0.088244
GOOG weight    0.617486
BP weight      0.001580
XOM weight     0.380934

Month 1 (Short)
Returns        0.003940
Volatility     0.062892
AAPL weight    0.085117
IBM weight     0.499776
COST weight    0.371804
GS weight      0.043303

        ...
        ...
        
Month 12 (Long)
Returns        0.002458
Volatility     0.022177
AAPL weight    0.113112
IBM weight     0.091965
GOOG weight    0.122320
BP weight      0.289261
XOM weight     0.106204
COST weight    0.075574
GS weight      0.201563

Month 12 (Short)
Returns       0.0
Volatility    0.0


'''        
        
        

'''----------------------------------------------------------------------------------------'''

'''Question 5'''
'''Clone the git repo and perform these command in git bash'''
#a
git ls-files "./*.py" | wc -l
'''
Output:
1
'''
#b
git ls-files | xargs cat | sed '/^\s*$/d' | wc -l
'''
Output:
265
'''
#c
git ls-files | xargs cat | grep -i -o '\<def\>' | wc -l
'''
Output:
9
'''
#d
git diff --shortstat cba196ffb88501aa2ce087f75e5847c828f08087 3ecf5d05bddc8ff990681a536d7dc48c7cdfa94e
'''
Output:
 1 file changed, 3 insertions(+), 84 deletions(-)
'''
#e
git clone git@github.com:samteo/my-python-project.git
def get_size():
    import os
    p=os.getcwd()
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
print(get_size())
'''
Output:
0.012322MB
'''
'''----------------------------------------------------------------------------------------'''

'''Question 6'''
import re
text = '2009/09/04 or 12/24/2009 or 04/12/2009 or 19 Jan 1992 or 19 Sept 1992 04/12/21 '
date_reg_exp1 = re.compile('\d{4}[/]\d{2}[/]\d{2}')
date_reg_exp2 = re.compile('\d{2}[/]\d{2}[/]\d{4}')
date_reg_exp3 = re.compile('\d{2}\s[A-Z]+[a-z]{2,3}\s\d{4}')
regex_list = [date_reg_exp1,date_reg_exp2,date_reg_exp3]
date=[]

for r in regex_list:
    matches_list=r.findall(text)
    for m in matches_list:
        date.append(m)

print('Number of appearance:',len(date))
'''
Output:
Number of appearance: 5    
'''






'''Question 6'''
import re
text = '01/01/2000'
date_reg_exp1 = re.compile(r"^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|(?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$")


regex_list = [date_reg_exp1]
date=[]

for r in regex_list:
    matches_list=r.findall(text)
    for m in matches_list:
        date.append(m)

print('Number of appearance:',len(date))





