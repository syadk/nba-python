import pandas as pd
import os



os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data')
###########################     2016-2017
df2016_dfs = pd.read_excel('NBA-2016-2017-DFS-Dataset.xlsx', header=0, skiprows=1)
df2016_dfs.columns = ['DATASET', 'DATE', 'PLAYER', 'OWN TEAM',
                      'OPPONENT TEAM', 'VENUE (R/H)', 'MINUTES', 
              'USAGE RATE',
              'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
              'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
              'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']
df2016_dfs.drop(columns= ['DATASET', 'OWN TEAM',
       'OPPONENT TEAM', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO', 
       'SAL - DRAFTKINGS', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - YAHOO'], inplace=True)

df2016_player = pd.read_excel('NBA-2016-2017-Player-BoxScore-Dataset.xlsx')
df2016_player.columns = ['DATASET', 'DATE',  'PLAYER',
       'POSITION', 'OWN TEAM', 'OPPONENT TEAM', 'VENUE (R/H)',
       'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR',
       'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS']
df2016_merged = df2016_player.merge(df2016_dfs, how='left', left_on=['DATE', 'PLAYER'],
                            right_on=['DATE', 'PLAYER'])
del df2016_dfs, df2016_player


##########################      2017-2018
df2017_dfs = pd.read_excel('NBA-2017-2018-DFS-Dataset.xlsx', header=0, skiprows=1)
df2017_dfs.columns = ['DATASET', 'DATE', 'PLAYER', 'OWN TEAM',
       'OPPONENT TEAM', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE',
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
       'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']
df2017_dfs.drop(columns= ['DATASET', 'OWN TEAM',
       'OPPONENT TEAM', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'POS - DRAFTKINGS', 'POS - FANDUEL',
       'POS - YAHOO', 'SAL - DRAFTKINGS','SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - YAHOO'], inplace=True)

df2017_player = pd.read_excel('NBA-2017-2018-Player-BoxScore-Dataset.xlsx')
df2017_player.columns = ['DATASET', 'DATE',  'PLAYER',
       'POSITION', 'OWN TEAM', 'OPPONENT TEAM', 'VENUE (R/H)',
       'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR',
       'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS']

df2017_merged = df2017_player.merge(df2017_dfs, how='left', left_on=['DATE', 'PLAYER'],
                            right_on=['DATE', 'PLAYER'])
del df2017_dfs, df2017_player

########################     2018-2019
df2018_dfs = pd.read_excel('NBA-2018-2019-DFS-Dataset.xlsx', header=1)
df2018_dfs.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER', 'OWN TEAM',
       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'DAYS REST', 
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
       'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']
df2018_dfs.drop(columns= ['DATASET', 'GAME-ID', 'PLAYER-ID', 'OWN TEAM',
       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'DAYS REST', 'POS - DRAFTKINGS', 'POS - FANDUEL',
       'POS - YAHOO', 'SAL - DRAFTKINGS','SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - YAHOO'], inplace=True)

df2018_player = pd.read_excel('NBA-2018-2019-Player-BoxScore-Dataset.xlsx')
df2018_player.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER',
       'POSITION', 'OWN TEAM', 'OPPONENT TEAM', 'VENUE (R/H)',
       'STARTER (Y/N)', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR',
       'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'USAGE RATE',
       'DAYS REST']
df2018_player.drop(columns = ['GAME-ID', 'PLAYER-ID', 'STARTER (Y/N)', 'USAGE RATE', 'DAYS REST'], inplace=True)

df2018_merged = df2018_player.merge(df2018_dfs, how='left', left_on=['DATE', 'PLAYER'],
                            right_on=['DATE', 'PLAYER'])

del df2018_dfs, df2018_player

######################       save files
df2016_merged.to_excel('NBA-2016-2017-Player-Boxscore-DFS_merged.xlsx')
df2017_merged.to_excel('NBA-2017-2018-Player-Boxscore-DFS_merged.xlsx')
df2018_merged.to_excel('NBA-2018-2019-Player-Boxscore-DFS_merged.xlsx')
#pickle
os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
df2016_merged.to_pickle('NBA-2016-2017-Player-Boxscore-DFS_merged.pkl')
df2017_merged.to_pickle('NBA-2017-2018-Player-Boxscore-DFS_merged.pkl')
df2018_merged.to_pickle('NBA-2018-2019-Player-Boxscore-DFS_merged.pkl')


