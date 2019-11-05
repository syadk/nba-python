import pandas as pd
import os

os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data')
#import the sample DFS which is from the 2018-2019 season
df = pd.read_excel('NBA-DFS-Sample-Dataset.xlsx', header=1)
#import the player stats from the 2018-2019 season
os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df_player = pd.read_pickle('NBA-2018-2019-Player-BoxScore-Dataset.pkl')


df.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER', 'OWN TEAM',
       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'DAYS REST', 
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
       'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']

df.drop(columns= ['DATASET', 'DATE', 'PLAYER', 'OWN TEAM',
       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'DAYS REST', 'POS - DRAFTKINGS', 'POS - FANDUEL',
       'POS - YAHOO', 'SAL - DRAFTKINGS','SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - YAHOO'], inplace=True)

df_player.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER FULL NAME',
       'POSITION', 'OWN TEAM', 'OPPONENT TEAM', 'VENUE (R/H)',
       'STARTER (Y/N)', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR',
       'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'USAGE RATE (%)',
       'DAYS REST']

df_merged = df_player.merge(df, how='left', left_on=['GAME-ID', 'PLAYER-ID'],
                            right_on=['GAME-ID', 'PLAYER-ID'])


os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df_merged.to_pickle('NBA-2018-2019-Player-Boxscore-DFS.pkl')
