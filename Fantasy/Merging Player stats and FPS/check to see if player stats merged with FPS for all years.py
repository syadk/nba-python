import pandas as pd
import os
os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')

df2016 = pd.read_pickle('NBA-2016-2017-Player-Boxscore-DFS_merged.pkl')
df2017 = pd.read_pickle('NBA-2017-2018-Player-Boxscore-DFS_merged.pkl')
df2018 = pd.read_pickle('NBA-2018-2019-Player-Boxscore-DFS_merged.pkl')


df2016.columns = ['DATA SET', 'DATE', 'PLAYER FULL NAME', 'POSITION', 'OWN TEAM',
       'OPP TEAM', 'VENUE (R/H)', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',
       'OR', 'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'SAL - FANDUEL',
       'FPS - FANDUEL']

df2017.columns = ['DATASET', 'DATE', 'PLAYER', 'POSITION', 'OWN TEAM', 
        'OPPONENT TEAM', 'VENUE (R/H)', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 
        'OR', 'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'SAL - FANDUEL',
       'FPS - FANDUEL']

df2018.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER FULL NAME', 'POSITION', 'OWN TEAM', 
       'OPPONENT TEAM', 'VENUE (R/H)', 'STARTER (Y/N)', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',
       'OR', 'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'USAGE RATE (%)', 'DAYS REST', 'SAL - FANDUEL', 
       'FPS - FANDUEL']

Gameid playerid starter usagerate daysrest