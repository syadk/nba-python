import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os 


for i in range(1,11)
    N=i
    
    def partTwo(GameID):
        N2=1
        threshold=5
        df_Game = df.loc[df['GameID'] == GameID]
        oppTeam=df_Game['OPP TEAM'].iloc[0]
        ownTeam=df_Game['OWN TEAM'].iloc[0]
        day=df_Game['DATE'].iloc[0]
        place=df_Game['VENUE (R/H)'].iloc[0]
        df_out=pd.DataFrame()        
    
        
    	 ################################################################################
    	 #Opposing team data
        df_Opp_Ag=df[df['OPP TEAM']==oppTeam]
        df_Opp_For=df[df['OWN TEAM']==oppTeam]
        
    	 #What was the date of the opposing teams last N
        oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
        if len(oppDates)-N>=0:
            dayNback=oppDates[len(oppDates)-N]
        
        	  #Only want data for the opposing teams previous N games
            df_Opp_Ag=df_Opp_Ag[(df_Opp_Ag['DATE']<day) & (df_Opp_Ag['DATE']>=dayNback)]
            df_Opp_For=df_Opp_For[(df_Opp_For['DATE']<day) & (df_Opp_For['DATE']>=dayNback)]
    
        	  #Calculate metrics for opposing team
            Opp_FGperc_Ag=df_Opp_Ag['FG'].sum()/df_Opp_Ag['FGA'].sum()
            Opp_FGperc_For=df_Opp_For['FG'].sum()/df_Opp_For['FGA'].sum()
            Opp_FG_Ag=df_Opp_Ag['FG'].sum()
            Opp_FG_For=df_Opp_For['FG'].sum()
            Opp_REB_For=df_Opp_For['TOT'].sum()
            Opp_REB_Ag=df_Opp_Ag['TOT'].sum()
    
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
                Own_FGperc_Ag=df_Own_Ag['FG'].sum()/df_Own_Ag['FGA'].sum()
                Own_FGperc_For=df_Own_For['FG'].sum()/df_Own_For['FGA'].sum()
                Own_FG_Ag=df_Own_Ag['FG'].sum()
                Own_FG_For=df_Own_For['FG'].sum()
    
    
                ##############################################################################
            	   #Go through each player
                for i in df_Game['PLAYER FULL NAME']:
                    df_Player=df[df['PLAYER FULL NAME']==i]
                    actual=df_Player[df_Player['DATE']==day]['TOT'].iloc[0] 
                    #Only want data before today to calculate historical statistics
                    df_Player=df_Player[df_Player['DATE']<day]
                    if df_Player.shape[0]>=N:
                        if sum(df_Player['TOT'].tail(N))>threshold*N:
                            #Calculate player statistics
                            Player_Rebounds=(df_Player['TOT'].tail(N)).sum()
                            Player_Rebounds_Short=(df_Player['TOT'].tail(N2)).sum()
                            DaysOff=day-df_Player['DATE'].iloc[-1]				
                            #Put result together with indicators
                            df_temp=pd.DataFrame([actual,Player_Rebounds,Player_Rebounds_Short,
                                                  Own_FGperc_Ag,Own_FGperc_For,Own_FG_Ag,Own_FG_For,
                                                  Opp_FGperc_Ag,Opp_FGperc_For,Opp_FG_Ag,Opp_FG_For,
                                                  Opp_REB_For,Opp_REB_Ag,
                                                  DaysOff,place,i,day]).transpose()
                            df_out=pd.concat([df_out, df_temp])
        return df_out
            
    
    # -------------------------------------------------------------------------------------------------------------------
    #os.chdir('C:\\Users\Kareem Kudus\Desktop\Python Stuff\Basketball')
    df_data = pd.DataFrame()    
    def f_gameID(x):  
        return x[1]+x[4]+x[5] 
    ###########################2016-2017###################################
    df = pd.read_pickle('C:/Users/syad/OneDrive/NBA Python/BigDataBall Data/2016-2017 NBA Player Boxscore.pkl')
    df['GameID']=df.apply(f_gameID, axis=1)
    df['DATE'] = pd.to_datetime(df['DATE'])
    # loop for every game
    uniqueGames = df['GameID'].unique()
    for i in uniqueGames:
        df_result = pd.DataFrame(partTwo(str(i)))
        df_data = pd.concat([df_data, df_result])
        print(i)
    
    ###########################2017-2018###################################
    df = pd.read_pickle('C:/Users/syad/OneDrive/NBA Python/BigDataBall Data/2017-2018 NBA Player Boxscore.pkl')
    df['GameID']=df.apply(f_gameID, axis=1)
    df['DATE'] = pd.to_datetime(df['DATE'])
    # loop for every game
    uniqueGames = df['GameID'].unique()
    for i in uniqueGames:
        df_result = pd.DataFrame(partTwo(str(i)))
        df_data = pd.concat([df_data, df_result])
        print(i)
          
        
        
    
    
    
        
    df_data.columns=['Actual','Player_Rebounds','Player_Rebounds_Short',
                                                  'Own_FGperc_Ag','Own_FGperc_For','Own_FG_Ag','Own_FG_For',
                                                  'Opp_FGperc_Ag','Opp_FGperc_For','Opp_FG_Ag','Opp_FG_For',
                                                  'Opp_REB_For','Opp_REB_Ag',
                                                  'DaysOff','place','player','day']
    # ------------------------
    
    
    def f_dates(x):  
        return x[13].days
    df_data['DaysOff']=df_data.apply(f_dates, axis=1)
    
    def f_place(x): 
        if x[14]=='R':
            return 0
        else:
            return 1
    df_data['place']=df_data.apply(f_place, axis=1)
    
    #df_data.to_pickle('RBS dataframe N1 and 10.pkl')
    df_data.to_pickle('RBS dataframe N'+str(N)+'.pkl')


