import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


os.chdir('C:\\GitHub\\nba-python\\Rebounds\\Accuracy of Pinnacle - Rebounds')
df = pd.read_excel('Actual Results - Rebounds - nov 25.xlsx')

#average error
df['error'] = df['#'] - df['Actual']
df['error squared'] = np.square(df['error'])
average_error_squared = df['error squared'].sum() / len(df)

######## implied probability vs actual probability ----------------------------
df['IP_over'] = (1/df['over_price']).round(decimals=2)
##0 if the actual result is under the mid, 1 if actual result is over the mid
def fn_binary(row):
    if (row['Actual'] > row['#']):
        return 1
    else:
        return 0        
df['over?'] = df.apply(fn_binary, axis=1)
## make a dictionary/dataframe whose keys are the bins (implied probabiity) and values are the actual probability
dict_bin = {}
df_graph = pd.DataFrame(columns=['IP_over', 'count','total', 'actual %'])
row = 0
for i in sorted(df['IP_over'].unique()):
    dftemp = df.loc[df['IP_over'] == i]
    total = len(dftemp)
    count = dftemp['over?'].sum()
    #dict_bin[i] = (count/total)
    dict_bin[i] = [count, total, (count/total)]
df_graph = pd.DataFrame.from_dict(dict_bin, orient='index')