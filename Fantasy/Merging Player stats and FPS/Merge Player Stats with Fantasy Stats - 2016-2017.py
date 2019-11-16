import pandas as pd
import os

os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data')
#import the DFS file for specific year
df = pd.read_excel('NBA-2016-2017-DFS-Dataset.xlsx', header=0, skiprows=1)
df.columns = ['DATASET', 'DATE', 'PLAYER FULL NAME', 'TEAM', 'OPPONENT', 'VENUE (R/H)',
              'MINUTES', 'USAGE RATE(%)',
              'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
              'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
              'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']
#import the player stats for specific year
#os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df_player = pd.read_excel('NBA-2016-2017-Player-BoxScore-Dataset.xlsx')

#
#df.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER', 'OWN TEAM',
#       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
#       'USAGE RATE', 'DAYS REST', 
#       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
#       'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
#       'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']

df.drop(columns= ['DATASET', 'TEAM',
       'OPPONENT', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE(%)', 
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO', 
       'SAL - DRAFTKINGS', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - YAHOO'], inplace=True)


df_merged = df_player.merge(df, how='left', left_on=['PLAYER FULL NAME', 'DATE'],
                            right_on=['PLAYER FULL NAME', 'DATE'])


os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df_merged.to_pickle('NBA-2016-2017-Player-Boxscore-DFS_merged.pkl')
