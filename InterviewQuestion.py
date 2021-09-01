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



