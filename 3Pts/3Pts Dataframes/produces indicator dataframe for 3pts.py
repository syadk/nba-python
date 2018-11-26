import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os 


N=10

def partTwo(GameID,df):
    N2=1
    threshold=0.5
    df_Game = df.loc[df['GameID'] == GameID]
    oppTeam=df_Game['OPP TEAM'].iloc[0]
    day=df_Game['DATE'].iloc[0]
    place=df_Game['VENUE (R/H)'].iloc[0]
    df_out=pd.DataFrame()        

    
	################################################################################
	#Opposing team data
    df_Opp_Ag=df[df['OPP TEAM']==oppTeam]
    
	#What was the date of the opposing teams last N
    oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
    if len(oppDates)-N>=0:
        dayNback=oppDates[len(oppDates)-N]
    
    	  #Only want data for the opposing teams previous N games
        df_Opp_Ag=df_Opp_Ag[(df_Opp_Ag['DATE']<day) & (df_Opp_Ag['DATE']>=dayNback)]
        #Need to fix the criteria here once date problem is fixed
    	  #Calculate metrics for opposing team
        Opp_3Pperc_Ag=df_Opp_Ag['3P'].sum()/df_Opp_Ag['3PA'].sum()
        Opp_3PT_Ag=df_Opp_Ag['3P'].sum()
        
        
    	  #################################################################################
    	  #Go through each player
        for i in df_Game['PLAYER FULL NAME']:
            df_Player=df[df['PLAYER FULL NAME']==i]
            actual=df_Player[df_Player['DATE']==day]['3P'].iloc[0] 
            #Only want data before today to calculate historical statistics
            df_Player=df_Player[df_Player['DATE']<day]
            if df_Player.shape[0]>=N:
                if sum(df_Player['3P'].tail(N))>threshold*N:
                    #Calculate player statistics
                    Player_3PT=(df_Player['3P'].tail(N)).sum()
                    Player_3PT_median=(df_Player['3P'].tail(N)).median()
                    Player_3PT_max=(df_Player['3P'].tail(N)).max()
                    Player_3PT_min=(df_Player['3P'].tail(N)).min()
                    Player_3PT_S=(df_Player['3P'].tail(N2)).sum()
                    Player_3PTperc=(df_Player['3P'].tail(N)).sum()/(df_Player['3PA'].tail(N)).sum()                
                    DaysOff=day-df_Player['DATE'].iloc[-1]				
                    #Put result together with indicators
                    df_temp=pd.DataFrame([actual,Player_3PT,Player_3PT_median,Player_3PT_max,Player_3PT_min,
                                          Player_3PT_S,Player_3PTperc,Opp_3PT_Ag,Opp_3Pperc_Ag,DaysOff,place,i,day]).transpose()
                    df_out=pd.concat([df_out, df_temp])
    return df_out
        

# -------------------------------------------------------------------------------------------------------------------
#os.chdir('C:\\Users\Kareem Kudus\Desktop\Python Stuff\Basketball')
df_data = pd.DataFrame()    
def f_gameID(x):  
    return x[1]+x[4]+x[5] 
###########################2016-2017###################################
df_main = pd.read_pickle('C:/Users/syad/OneDrive/NBA Python/BigDataBall Data/2016-2017 NBA Player Boxscore.pkl')

df_main['GameID']=df_main.apply(f_gameID, axis=1)
df_main['DATE'] = pd.to_datetime(df_main['DATE'])
# loop for every game
uniqueGames = df_main['GameID'].unique()
for i in uniqueGames:
    df_result = pd.DataFrame(partTwo(str(i), df_main))
    df_data = pd.concat([df_data, df_result])
    print(i)

###########################2017-2018###################################
df_main = pd.read_pickle('C:/Users/syad/OneDrive/NBA Python/BigDataBall Data/2017-2018 NBA Player Boxscore.pkl')
df_main['GameID']=df_main.apply(f_gameID, axis=1)
df_main['DATE'] = pd.to_datetime(df_main['DATE'])
# loop for every game
uniqueGames = df_main['GameID'].unique()
for i in uniqueGames:
    df_result = pd.DataFrame(partTwo(str(i),df_main))
    df_data = pd.concat([df_data, df_result])
    print(i)
    
    

    
df_data.columns=['Actual','Player_3PT','Player_3PT_median','Player_3PT_max','Player_3PT_min',
                 'Player_3PT_S','Player_3PTperc',
                 'Opp_3PT_Ag','Opp_3Pperc_Ag',
                 'DaysOff','place','player','day']
# ------------------------


def f_dates(x):  
    return x[9].days
df_data['DaysOff']=df_data.apply(f_dates, axis=1)

def f_place(x): 
    if x[10]=='R':
        return 0
    else:
        return 1
df_data['place']=df_data.apply(f_place, axis=1)

df_data.to_pickle('3Pt dataframe N'+str(N)+'.pkl')


