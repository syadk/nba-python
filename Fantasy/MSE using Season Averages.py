import pandas as pd
import os
import sklearn as sk



os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
###########################2016-2017###################################
df2017 = pd.read_pickle('NBA-2016-2017-Player-Boxscore-DFS_merged.pkl')
df2017.rename(columns={'FPS - DRAFTKINGS':'FPS','OPPONENT TEAM':'OPP TEAM', 'PLAYER':'PLAYER FULL NAME'},
          inplace=True)
df2017.drop(columns=['SAL - DRAFTKINGS', 'SAL - FANDUEL', 'FPS - FANDUEL'], inplace=True)

#create the average points column for each player
df_data = pd.DataFrame()
for i in df2017['PLAYER FULL NAME'].unique():
    df_player = df2017[df2017['PLAYER FULL NAME'] == i].copy()
    df_player['Avg_FPS'] = df_player['FPS'].expanding().mean()
    df_player['Avg_FPS'] = df_player['Avg_FPS'].shift(1)
    df_player = df_player[1:]
    df_data = pd.concat([df_data, df_player])

#calculate the MSE
MSE = sk.metrics.mean_squared_error(df_data['FPS'], df_data['Avg_FPS'])
print(MSE)