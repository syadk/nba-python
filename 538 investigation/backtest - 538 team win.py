import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from random import shuffle

os.chdir('C:\\GitHub\\nba-python\\538 investigation')


df_odds = pd.read_pickle('nba-season-team-feed.pkl')
df_odds = df_odds[['DATE','TEAM','VENUE','F','OPENING SPREAD', 'MONEYLINE']]
df_odds['DATE'] = pd.to_datetime(df_odds['DATE'])
#df_odds = df_odds.loc[df_odds['DATE'] < datetime.strptime('2018 04 15', '%Y %m %d')]

#convert to decmial odds-------------------------------------------------------
#df_odds['MONEYLINE'] = -110
def f_convert_odds(value):
    if (value > 0):
        x = (value + 100)/100
    elif (value < 0):
        x = ((-value)+100)/(-value)
    return x
df_odds['MONEYLINE'] = df_odds['MONEYLINE'].apply(f_convert_odds)


df_538 = pd.read_pickle('538 predictions for backtest.pkl')

df_538 = df_538.merge(df_odds, how='left', on=['DATE', 'TEAM'])

df = df_538[['DATE', 'TEAM', '538 predicted win %', 'F_x', 'win', 'VENUE',
       'MONEYLINE']]

df.columns = ['Date', 'Team', '538 predicted win %', 'Score', 'Win', 'Venue',
       'Moneyline']

df_expected=pd.DataFrame()

for i in range(0, df.shape[0]):
    yHat = df['538 predicted win %'].iloc[i]
    Moneyline = df['Moneyline'].iloc[i]
    today =  df['Date'].iloc[i]
    Team = df['Team'].iloc[i]
    Win = df['Win'].iloc[i]
    print(today)
    
    expectedReturn = (yHat*Moneyline)-1
    


    if (expectedReturn > 0):
        bet = True
    else:
        bet = False


    if (Win == 1) and (bet == True):
            winBet=True
    elif (Win == 0) and (bet == True):
            winBet=False
    else:
            winBet=False


    if (bet == True):
        if (winBet == True):
            money = Moneyline-1
        else:
            money = -1
    else:
        money = 0
#         
    temp=[today,Team, expectedReturn,
          yHat, bet, winBet, Win, money]
           
    
    
    thisOne=pd.DataFrame(temp)
    thisOne=thisOne.T
    df_expected=pd.concat([df_expected, thisOne],axis=0)

df_expected.columns = ['today','Team', 'expectedReturn',
          'yHat', 'bet', 'winBet', 'Team win', 'money'] 
df_expected.reset_index(drop=True,inplace=True)
df_expected['money'].sum()
print(np.sum(df_expected['money']))
df_analysis = df_expected.loc[df_expected['money'] != 0]
print(df_expected['winBet'].sum())
print(df_analysis.shape[0])
var = df_analysis['money'].cumsum()
