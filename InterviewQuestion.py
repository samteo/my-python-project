# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:14:14 2021

@author: samte
"""
'Question 1'
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
#Ans:
# Employee Dan salary is higher than his manager
# Employee Sally salary is higher than his manager
# Employee Joe salary is higher than his manager

#b
manager = set(manager)
not_manager=set(t.index).difference(manager)
avg_salary = t[t.index.isin(not_manager)]['Salary'].mean()
print(avg_salary)
#Ans:
# 425.0

'----------------------------------------------------------------------------------------'

'Question 2'
def exist():
    if 'v' in globals():
        return print('v is defined')
    else:
        return print('v is not defined')        
exist()

'----------------------------------------------------------------------------------------'

'Question 3'
def pascals_triangle(n_layers):
    result=[]
    for n in range(n_layers):
        row=[1]
        if result:
            last_row=result[-1]
            row.extend([sum(p) for p in zip(last_row,last_row[1:])])
            row.append(1)
        result.append(row)
        print(row)
           
pascals_triangle(6)
   
'----------------------------------------------------------------------------------------'

'Question 5'
#a
git ls-files "./*.py" | wc -l
#b
git ls-files | xargs cat | sed '/^\s*$/d' | wc -l
#c
git ls-files | xargs cat | grep -i -c def | wc -w
#d
git diff --shortstat cba196ffb88501aa2ce087f75e5847c828f08087 3ecf5d05bddc8ff990681a536d7dc48c7cdfa94e
#e
git clone git@github.com:samteo/my-python-project.git
def get_size():
    import os
    p=os.getcwd()+'\\repo\\firstfolder'  #'\\my-python-project'
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

'----------------------------------------------------------------------------------------'

'Question 6'
import re
text = '2009/09/04 or 12/24/2009 or 04/12/2009 or 19 Jan 1992 or 19 Sept 1992'
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













