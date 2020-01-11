import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np




N=6
 
def partTwo(GameID):
    
    threshold=3
    df_Game = df.loc[df['GAME-ID'] == GameID]
    oppTeam=df_Game['OPP TEAM'].iloc[0]
    ownTeam=df_Game['OWN TEAM'].iloc[0]
    day=df_Game['DATE'].iloc[0]
    place=df_Game['VENUE (R/H)'].iloc[0]
    df_out=pd.DataFrame()        

    
	 ################################################################################
	 #Opposing team data
     #games where the opposing team was the opposing team
    df_Opp_Ag=df[df['OPP TEAM']==oppTeam]
    #games where the opposing team was the subject team
    df_Opp_For=df[df['OWN TEAM']==oppTeam]
    
	 #What was the date of the opposing teams last N
    oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
    if len(oppDates)-N>=0:
        dayNback=oppDates[len(oppDates)-N]
    
    	  #Only want data for the opposing teams previous N games
        df_Opp_Ag=df_Opp_Ag[(df_Opp_Ag['DATE']<day) & (df_Opp_Ag['DATE']>=dayNback)]
        df_Opp_For=df_Opp_For[(df_Opp_For['DATE']<day) & (df_Opp_For['DATE']>=dayNback)]

    	#Calculate metrics for opposing team
        Opp_FPS_Ag=df_Opp_Ag['FPS'].sum()
        Opp_FPS_For=df_Opp_For['FPS'].sum()
        

        ################################################################################
        #Own team data
        df_Own_Ag=df[df['OPP TEAM']==ownTeam]
        df_Own_For=df[df['OWN TEAM']==ownTeam]
        #What was the date of the own teams last N
        ownDates=(df_Own_For[df_Own_For['DATE']<day])['DATE'].unique()
        if len(ownDates)-N>=0:
            dayNback=ownDates[len(ownDates)-N]
            #Only want data for the opposing teams previous N games
            df_Own_Ag=df_Own_Ag[(df_Own_Ag['DATE']<day) & (df_Own_Ag['DATE']>=dayNback)]
            df_Own_For=df_Own_For[(df_Own_For['DATE']<day) & (df_Own_For['DATE']>=dayNback)]

            #Calculate metrics for own team
            Own_FPS_Ag=df_Own_Ag['FPS'].sum()
            Own_FPS_For=df_Own_For['FPS'].sum()
            
            


            ##############################################################################
        	   #Go through each player
            for i in df_Game['PLAYER FULL NAME']:
                df_Player=df[df['PLAYER FULL NAME']==i]
                actual=df_Player[df_Player['DATE']==day]['FPS'].iloc[0] 
                #Only want data before today to calculate historical statistics
                df_Player=df_Player[df_Player['DATE']<day]
                if df_Player.shape[0]>=N:
                    if sum(df_Player['FPS'].tail(N))>threshold*N:
                        #Calculate player statistics
                        Player_FPS=((df_Player['FPS'].tail(N)).sum())/N
                        Player_FPS_1 = (df_Player['FPS'].tail(1).sum())/1
                        Player_FPS_2 = (df_Player['FPS'].tail(2).sum())/2
                        Player_FPS_3 = (df_Player['FPS'].tail(3).sum())/3
                        Player_FPS_4 = (df_Player['FPS'].tail(4).sum())/4
                        Player_FPS_5 = (df_Player['FPS'].tail(5).sum())/5
                        Player_FPS_6 = (df_Player['FPS'].tail(6).sum())/6
                        Player_FPS_7 = (df_Player['FPS'].tail(7).sum())/7
                        Player_FPS_8 = (df_Player['FPS'].tail(8).sum())/8
                        Player_FPS_9 = (df_Player['FPS'].tail(9).sum())/9
                        Player_FPS_10 = (df_Player['FPS'].tail(10).sum())/10                       
                        DaysOff=day-df_Player['DATE'].iloc[-1]
                        Median_FPS=df_Player['FPS'].tail(N).median()
                        Std_FPS=df_Player['FPS'].tail(N).std()
                        Max_FPS=df_Player['FPS'].tail(N).max()-((df_Player['FPS'].tail(N)).sum())/N
                        Min_FPS=df_Player['FPS'].tail(N).min()-((df_Player['FPS'].tail(N)).sum())/N
                        Player_Points = df_Player['PTS'].tail(N).sum()/N
                        Player_TOT = df_Player['TOT'].tail(N).sum()/N
                        Player_Minutes = df_Player['MIN'].tail(N).sum()/N
                        Player_FG = df_Player['FG'].tail(N).sum()/N
                        Player_FGA = df_Player['FGA'].tail(N).sum()/N
                        Player_3P = df_Player['3P'].tail(N).sum()/N
                        Player_3PA = df_Player['3PA'].tail(N).sum()/N
                        Player_Assists = df_Player['A'].tail(N).sum()/N
                        Player_Fouls = df_Player['PF'].tail(N).sum()/N
                        Player_Turnovers = df_Player['TO'].tail(N).sum()/N
                        Player_Blocks = df_Player['BL'].tail(N).sum()/N
                        #think about putting salary, prob not though since will heavily influence model with their predictions
                        #Put result together with indicators
                        global var_list

                        var_list = {'actual':actual, 'Player_FPS':Player_FPS, 
                                    'Player_FPS_1':Player_FPS_1, 'Player_FPS_2':Player_FPS_2, 'Player_FPS_3':Player_FPS_3,
                                    'Player_FPS_4':Player_FPS_4, 'Player_FPS_5':Player_FPS_5, 'Player_FPS_6':Player_FPS_6,
                                    'Player_FPS_7':Player_FPS_7, 'Player_FPS_8':Player_FPS_8, 'Player_FPS_9':Player_FPS_9,
                                    'Player_FPS_10':Player_FPS_10,
                                              'Median_FPS':Median_FPS, 'Std_FPS':Std_FPS, 'Max_FPS':Max_FPS,
                                              'Min_FPS':Min_FPS, 'Player_Points':Player_Points, 'Player_TOT':Player_TOT,
                                              'Player_Minutes':Player_Minutes, 
                                              'Player_FG':Player_FG, 'Player_FGA':Player_FGA,
                                              'Player_3P':Player_3P, 'Player_3PA':Player_3PA,
                                              'Player_Assists':Player_Assists, 'Player_Fouls':Player_Fouls,
                                              'Player_Turnovers':Player_Turnovers, 'Player_Blocks':Player_Blocks,
                                              'Opp_FPS_Ag':Opp_FPS_Ag, 'Opp_FPS_For':Opp_FPS_For,
                                              'Own_FPS_Ag':Own_FPS_Ag, 'Own_FPS_For':Own_FPS_For,
                                              'Own_Team':ownTeam, 'Opp_Team':oppTeam,
                                              'DaysOff':DaysOff,'place':place,'Player':i,'day':day}
                        df_temp=pd.DataFrame.from_dict(var_list, orient='index').transpose()                       
                        df_out=pd.concat([df_out, df_temp])
    return df_out
        

# -------------------------------------------------------------------------------------------------------------------
df_data = pd.DataFrame()    
def f_gameID(x):  
    return x[1]+x[4]+x[5] 

os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data\Pickle')
###########################2016-2017###################################
df2017 = pd.read_pickle('NBA-2016-2017-Player-Boxscore-DFS_merged.pkl')
############################2017-2018###################################
df2018 = pd.read_pickle('NBA-2017-2018-Player-Boxscore-DFS_merged.pkl')
###########################2018-2019 ###################################
df2019 = pd.read_pickle('NBA-2018-2019-Player-Boxscore-DFS_merged.pkl')


######## MERGE ALL 3 SEASONS
df = pd.concat([df2017,df2018,df2019])
df['GAME-ID']=df.apply(f_gameID, axis=1)
df['DATE'] = pd.to_datetime(df['DATE'])
df.rename(columns={'FPS - DRAFTKINGS':'FPS','OPPONENT TEAM':'OPP TEAM', 'PLAYER':'PLAYER FULL NAME'},
          inplace=True)


# loop for every game
uniqueGames = df['GAME-ID'].unique()
for i in uniqueGames:
    df_result = pd.DataFrame(partTwo((i)))
    df_data = pd.concat([df_data, df_result])
    print(i)
  
df_data.columns = list(var_list.keys())
# ------------------------


def f_dates(x):  
    return x['DaysOff'].days
df_data['DaysOff']=df_data.apply(f_dates, axis=1)

def f_place(x): 
    if x['place']=='R':
        return 0
    else:
        return 1
df_data['place']=df_data.apply(f_place, axis=1)

#####  ADD DUMMY FOR EACH PLAYER
df_dummies = pd.get_dummies(df_data['Player'])
dfnew = pd.concat([df_data,df_dummies], axis=1)




#df_data.to_pickle('RBS dataframe N'+str(N)+'.pkl')
os.chdir('C:\\GitHub\\nba-python\\Fantasy\\FPS Input Data for Model')
file_name = 'DFS Input_N'+str(N)+'.pkl'
df_data.to_pickle(file_name)
#file_name2 = 'DFS Input with Dummies_N'+str(N)+'.pkl'
#dfnew.to_pickle(file_name2)



