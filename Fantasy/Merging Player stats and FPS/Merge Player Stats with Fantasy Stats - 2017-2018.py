import pandas as pd
import os

os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data')
#import the DFS file for specific year
df = pd.read_excel('NBA-2017-2018-DFS-Dataset.xlsx', header=0, skiprows=1)
#import the player stats for specific year
os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df_player = pd.read_pickle('NBA-2017-2018-Player-BoxScore-Dataset.pkl')


df.columns = ['DATASET', 'DATE', 'PLAYER', 'OWN TEAM',
       'OPPONENT TEAM', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE',
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
       'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']

df.drop(columns= ['DATASET', 'OWN TEAM',
       'OPPONENT TEAM', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'POS - DRAFTKINGS', 'POS - FANDUEL',
       'POS - YAHOO', 'SAL - DRAFTKINGS','SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - YAHOO'], inplace=True)

df_player.columns = ['DATASET', 'DATE',  'PLAYER',
       'POSITION', 'OWN TEAM', 'OPPONENT TEAM', 'VENUE (R/H)',
       'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR',
       'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS']

df_merged = df_player.merge(df, how='left', left_on=['DATE', 'PLAYER'],
                            right_on=['DATE', 'PLAYER'])


os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df_merged.to_pickle('NBA-2017-2018-Player-Boxscore-DFS_merged.pkl')
