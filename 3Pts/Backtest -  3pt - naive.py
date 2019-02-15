# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:07:29 2018

@author: Kareem Kudus
"""
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

#######################################User defined variables#####################################
#os.chdir('C:\\Users\Kareem Kudus\Desktop\Python Stuff\Basketball')
stat='3Pt'
N=5 #THis shouldnt really beuser defined - should pull in from optimal parameters
historical='Player_3PT'
N2=1
#Will need to generalzie to all mids later
mid=1.5

    

##########################################Pull in odds###########################################
df_odds = pd.read_excel("C:/GitHub/nba-python/Player Specials with Actual - dec 27.xlsx")
df_odds=df_odds[df_odds['units']=='ThreePointFieldGoals']
df_odds = df_odds.dropna()
df_odds=df_odds[df_odds['#']==mid] #This is a simplifying assumption, for now
df_odds=df_odds.reset_index()   
df_odds['dateGame'] = pd.to_datetime(df_odds['dateGame'])                      

#########################################Data from this season###################################
df_this_season = pd.read_excel("C:/GitHub/nba-python/nba-season-player-feed.xlsx")
df_this_season['DATE'] = pd.to_datetime(df_this_season['DATE'])


#DO THIS WHOLE THING FOR EACH MID?
#Can probably find optimized parameters for each midfor threepts though since there are only a few mids

####################################Optimal parameters##########################################
#It might be a bad assumption to use the same parameters for each mid, but do we have another option?
#It isnt really practical to find optimal parameters for each mid?
df_parameters = pd.read_csv("3Pt results.csv")
df_parameters =df_parameters[df_parameters['mid']==mid]
#Need to also filter based on N later
df_parameters =df_parameters.sort_values(by=['Boost test'])
#Now set parameters to their optimized values
opt_n_estimators=df_parameters['opt_n_estimators'].iloc[0]
opt_learning_rate=df_parameters['opt_learning_rate'].iloc[0]
opt_max_depth=df_parameters['opt_max_depth'].iloc[0]
opt_min_samples_leaf=df_parameters['opt_min_samples_leaf'].iloc[0]
opt_min_split=df_parameters['opt_min_split'].iloc[0]
upper=df_parameters['upper'].iloc[0]
lower=df_parameters['lower'].iloc[0]


#Come up estimates of over/under for each bet - do this seperately for each mid
#Do for just one mid for now

#Load up training data
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df = df.drop(['player', 'day'], axis=1)
#Adjustments based on this mid
df=df[(df[historical]<upper*N) & (df[historical]>lower*N)]
df.loc[df['Actual'] < mid,['Actual']]= 0
df.loc[df['Actual'] > mid,['Actual']]= 1
trainy = pd.to_numeric(df['Actual'])
trainx = (df.drop(['Actual'], axis=1)).values
#Train model
model = GradientBoostingClassifier(n_estimators=opt_n_estimators,min_samples_split=opt_min_split,
                                max_depth=opt_max_depth,learning_rate=opt_learning_rate,
                                min_samples_leaf=opt_min_samples_leaf)
model.fit(trainx, trainy)

df_bets=pd.DataFrame()
#Now for each potential bet
for i in range(df_odds.shape[0]):
    df_bet=df_odds.ix[i]
    o=df_bet['over_price']
    u=df_bet['under_price']
    overIP=1/o
    underIP=1/u
    player=df_bet['New Name']
    day=df_bet['dateGame']
    df_game=df_this_season[(df_this_season['PLAYER \nFULL NAME']==player)&(df_this_season['DATE']==day)]
    oppTeam=df_game['OPPONENT \nTEAM'].iloc[0] 
    place=df_game['VENUE\n(R/H)'].iloc[0]
    if place=='R':
        place=0
    else:
        place=1
    #Calculate indicators for this bet
    ################################################################################
    #Opposing team data
    df_Opp_Ag=df_this_season[df_this_season['OPPONENT \nTEAM']==oppTeam]
    oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
    if len(oppDates)-N>=0:
        dayNback=oppDates[len(oppDates)-N]
        #Only want data for the opposing teams previous N games
        df_Opp_Ag=df_Opp_Ag[(df_Opp_Ag['DATE']<day) & (df_Opp_Ag['DATE']>=dayNback)]
        #Need to fix the criteria here once date problem is fixed
    	  #Calculate metrics for opposing team
        Opp_3Pperc_Ag=df_Opp_Ag['3P'].sum()/df_Opp_Ag['3PA'].sum()
        Opp_3PT_Ag=df_Opp_Ag['3P'].sum()
        df_Player=df_this_season[df_this_season['PLAYER \nFULL NAME']==player]
        actual=df_Player[df_Player['DATE']==day]['3P'].iloc[0]
        df_Player=df_Player[df_Player['DATE']<day]
        if df_Player.shape[0]>=N:
            Player_3PT=(df_Player['3P'].tail(N)).sum()
            Player_3PT_median=(df_Player['3P'].tail(N)).median()
            Player_3PT_max=(df_Player['3P'].tail(N)).max()
            Player_3PT_min=(df_Player['3P'].tail(N)).min()
            Player_3PT_S=(df_Player['3P'].tail(N2)).sum()
            Player_3PTperc=(df_Player['3P'].tail(N)).sum()/(df_Player['3PA'].tail(N)).sum()
            DaysOff=(day-df_Player['DATE'].iloc[-1]).days
            predictionInput=np.array([Player_3PT,Player_3PT_median,Player_3PT_max,Player_3PT_min,
                                          Player_3PT_S,Player_3PTperc,Opp_3PT_Ag,Opp_3Pperc_Ag,DaysOff,place])
            predictionInput = predictionInput.reshape(1,-1)
            yHat= model.predict_proba(predictionInput)
            overPredictedIP=yHat[0,1]
            underPredictedIP=yHat[0,0]

            
            returnO=yHat[0,1]*(o-1)-yHat[0,0]
            returnU=-yHat[0,1]-yHat[0,0]*(u-1)
            if (returnU > 0.05):
                under = True
                over = False
            elif (returnO > 0.05):
                under = False
                over  = True
            else:
                under = False
                over = False  
                
            diff=actual-mid
        
            if (diff>0) and (over==True):
                win=True
            elif (diff<0) and (under==True):
                win=True
            else:
                win=False 
                
            if (under == True):
                if (win == True):
                    money = u-1
                else:
                    money = -1
            elif (over == True):
                if (win == True):
                    money = o-1
                else:
                    money = -1
            else:
                money = 0
                
            #Make prediction for 0 and 1 for this bet
            df_temp=pd.DataFrame([player, day,over, under,o,u, mid,actual,overIP,overPredictedIP,underIP,underPredictedIP,returnO, returnU, win, money]).transpose()
            df_bets=pd.concat([df_bets, df_temp])
            
df_bets.columns=['player', 'day','over','under','o','u', 'mid','actual','overIP','overPredictedIP','underIP','underPredictedIP','returnO', 'returnU', 'win', 'money']    
df_analysis = df_bets.loc[df_bets['money'] != 0]
print(df_bets['win'].sum())
print(df_analysis.shape[0])
var = df_analysis['money'].cumsum()