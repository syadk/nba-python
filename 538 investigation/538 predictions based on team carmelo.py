import pandas as pd
import os

os.chdir('C:\\GitHub\\nba-python\\538 investigation')

df = pd.read_excel('predicted win percentage data based on carmelo as of jan 27, 2019.xlsx', header=None)

df = df.dropna(how='all')
df[0] = df[0].fillna(method='ffill')
df = df.dropna(thresh=2)
df.reset_index(drop=True, inplace=True)
#cavs vs bulls, kinds vs clippers
df = df[df.index % 3 != 0]

df = df.loc[df[1] != 'FINAL']
df = df.drop(columns=[1])

df1 = df[::2]
df1.reset_index(drop=True, inplace=True)
df2 = df[1::2]
df2.reset_index(drop=True, inplace=True)


df = pd.concat([df1,df2], axis=1)
df.columns = ['date1', 'team1', 'misc1', 'win% team1', 'score team1',
                     'date2', 'team2', 'misc2', 'win% team2', 'score team2']
#drop some columns we might need in the future
df = df[['date1', 'team1','win% team1', 'score team1',
                     'team2', 'win% team2', 'score team2']]
#######

def f_winner_df(row):
    if row['score team1'] > row['score team2']:
        win = 1
    else:
        win = 0
    return win
df['team1 win'] = df.apply(f_winner_df, axis=1)

df['team2 win'] = 1 - df['team1 win']

#was it a close game?
def close_game(row):
    if ((row['score team1'] > row['score team2']) and ((row['score team1']-5) < row['score team2'])):
        close = 1
    elif ((row['score team2'] > row['score team1']) and ((row['score team2']-5) < row['score team1'])):
        close = 1
    else:
        close = 0
    return close
df['Close Game'] = df.apply(close_game, axis=1)