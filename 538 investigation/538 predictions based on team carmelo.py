import pandas as pd
import os
import datetime

os.chdir('C:\\GitHub\\nba-python\\538 investigation')

df = pd.read_excel('predicted win percentage data based on carmelo as of jan 27, 2019.xlsx', header=None)

df = df.dropna(how='all')
df[0] = df[0].fillna(method='ffill')
df = df.dropna(thresh=2)
df.reset_index(drop=True, inplace=True)
#cavs vs bulls, kinds vs clippers
df = df[df.index % 3 != 0]

df = df.loc[df[1] != 'FINAL']
df = df.drop(columns=[1,3])
df['win'] = 0

#winner
for i in range(len(df)):
    if (i % 2 == 0): #even
        if (df[5].iloc[i] > df[5].iloc[i+1]):
            df['win'].iloc[i] = 1
    else:
        if (df[5].iloc[i] > df[5].iloc[i-1]):
            df['win'].iloc[i] = 1
            
df.columns = ['DATE', 'TEAM', '538 predicted win %', 'F', 'win']


def fn_fix_date(row):
    string = row['DATE'].split()
    month = string[1][:-1]
    if (month == 'Oct') or (month == 'Nov') or (month == 'Dec'):
        year = '2018'
    else:
        year = '2019'
    day = string[2]
    date_full = day + month + year
    date_full = datetime.datetime.strptime(date_full, '%d%b%Y')
    return(date_full)

df['DATE'] = df.apply(fn_fix_date, axis=1)

def fn_fix_team(row):
    team = row['TEAM']
    if team == 'Hawks':
        team = 'Atlanta'
    elif team == 'Celtics':
        team = 'Boston'
    elif team == 'Nets':
        team = 'Brooklyn'
    elif team == 'Hornets':
        team = 'Charlotte'
    elif team == 'Bulls':
        team = 'Chicago'
    elif team == 'Cavaliers':
        team = 'Cleveland'
    elif team == 'Mavericks':
        team = 'Dallas'
    elif team == 'Nuggets':
        team = 'Denver'
    elif team == 'Pistons':
        team = 'Detroit'
    elif team == 'Warriors':
        team = 'Golden State'
    elif team == 'Rockets':
        team = 'Houston'
    elif team == 'Clippers':
        team = 'LA Clippers'
    elif team == 'Lakers':
        team = 'LA Lakers'
    elif team == 'Grizzlies':
        team = 'Memphis'
    elif team == 'Heat':
        team = 'Miami'
    elif team == 'Bucks':
        team = 'Milwaukee'
    elif ((team == "T'wolves") or (team == 'Timberwolves')):
        team = 'Minnesota'
    elif team == 'Pelicans':
        team = 'New Orleans'
    elif team == 'Knicks':
        team = 'New York'
    elif team == 'Thunder':
        team = 'Oklahoma City'
    elif team == 'Magic':
        team = 'Orlando'
    elif team == '76ers':
        team = 'Philadelphia'
    elif team == 'Suns':
        team = 'Phoenix'
    elif team == 'Trail Blazers':
        team = 'Portland'
    elif team == 'Kings':
        team = 'Sacramento'
    elif team == 'Spurs':
        team = 'San Antonio'
    elif team == 'Raptors':
        team = 'Toronto'
    elif team == 'Jazz':
        team = 'Utah'
    elif team == 'Wizards':
        team = 'Washington'
    elif team == 'Pacers':
        team = 'Indiana'
    return team
df['TEAM'] = df.apply(fn_fix_team, axis=1)






df.to_pickle('538 predictions for backtest.pkl')

#df1 = df[::2]
#df1.reset_index(drop=True, inplace=True)
#df2 = df[1::2]
#df2.reset_index(drop=True, inplace=True)




#df = pd.concat([df1,df2], axis=1)
#df.columns = ['date1', 'team1', 'misc1', 'win% team1', 'score team1',
#                     'date2', 'team2', 'misc2', 'win% team2', 'score team2']
##drop some columns we might need in the future
#df = df[['date1', 'team1','win% team1', 'score team1',
#                     'team2', 'win% team2', 'score team2']]
########
#
#def f_winner_df(row):
#    if row['score team1'] > row['score team2']:
#        win = 1
#    else:
#        win = 0
#    return win
#df['team1 win'] = df.apply(f_winner_df, axis=1)
#
#df['team2 win'] = 1 - df['team1 win']
#
##was it a close game?
#def close_game(row):
#    if ((row['score team1'] > row['score team2']) and ((row['score team1']-5) < row['score team2'])):
#        close = 1
#    elif ((row['score team2'] > row['score team1']) and ((row['score team2']-5) < row['score team1'])):
#        close = 1
#    else:
#        close = 0
#    return close
#df['Close Game'] = df.apply(close_game, axis=1)