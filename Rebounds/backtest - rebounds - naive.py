# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:07:29 2018

@author: Kareem Kudus
"""
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE

#######################################User defined variables#####################################
#os.chdir('C:\\Users\Kareem Kudus\Desktop\Python Stuff\Basketball')
stat='RBS'
N=10 #THis shouldnt really beuser defined - should pull in from optimal parameters
#historical='Player_3PT'
N2=1
#Will need to generalzie to all mids later

    

##########################################Pull in odds###########################################
df_odds = pd.read_excel("C:/GitHub/nba-python/Player Specials with Actual - dec 27.xlsx")
df_odds=df_odds[df_odds['units']=='Rebounds']
df_odds = df_odds.dropna()
#df_odds=df_odds[df_odds['#']==mid] #This is a simplifying assumption, for now
df_odds=df_odds.reset_index(drop=True)   
df_odds['dateGame'] = pd.to_datetime(df_odds['dateGame'])                      

#########################################Data from this season###################################
df_this_season = pd.read_excel("C:/GitHub/nba-python/nba-season-player-feed.xlsx")
df_this_season['DATE'] = pd.to_datetime(df_this_season['DATE'])


#DO THIS WHOLE THING FOR EACH MID?
#Can probably find optimized parameters for each midfor threepts though since there are only a few mids

####################################Optimal parameters##########################################

#Come up estimates of over/under for each bet - do this seperately for each mid
#Do for just one mid for now

#Load up training data
dataFile=stat+' dataframe N'+str(N)+'.pkl'


#train model given a specific mid
def model_selection(mid):
    df = pd.read_pickle('C:/GitHub/nba-python/Rebounds/Rebounds Dataframes/'+dataFile)
    df = df.dropna()
    df = df.drop(['player', 'day'], axis=1)
    df = df[['Actual', 'Player_Rebounds', 'Own_FGperc_Ag', 'Own_FGperc_For',
         'Opp_FGperc_Ag', 'Opp_FGperc_For', 'place']]
    dftrain = df.copy()
    
    kmeans = KMeans(n_clusters=3, random_state=101)
    kmeans.fit(trainx)
    

    dftrain['cluster'] = kmeans.labels_
    
    dftrain1 = dftrain.loc[dftrain['cluster'] == 0]
    dftrain1 = dftrain1.drop(['cluster'], axis=1)
    dftrain1.loc[dftrain1['Actual'] < mid, ['Actual']] = 0
    dftrain1.loc[dftrain1['Actual'] > mid, ['Actual']] = 1
    dftrain1.reset_index(drop=True,inplace=True)
    trainx1 = dftrain1.drop(['Actual'], axis=1)
    trainy1 = pd.to_numeric(dftrain1['Actual'])
    

#    dftrain = dftrain.loc[dftrain['#'] == mid]



    
#    trainy = pd.to_numeric(dftrain['Actual'])
#    temp=(dftrain1.drop(['Actual'], axis=1))
#    temp.reset_index(drop=True,inplace=True)
#    trainx = temp.values
    mlp = LogisticRegression()
    mlp.fit(trainx1,trainy1)
    
    return(mlp)
    
def model_selection_regressor(mid):
    df = pd.read_pickle('C:/GitHub/nba-python/Rebounds/Rebounds Dataframes/'+dataFile)
    df = df.dropna()
    df = df.drop(['player', 'day'], axis=1)
#    df = df[['Actual', 'Player_Rebounds']]

    dftrain = df.copy()
#    dftrain = dftrain.loc[dftrain['#'] == mid]
 
    dftrain.reset_index(drop=True,inplace=True)

    
    trainy = pd.to_numeric(dftrain['Actual'])
    temp=(dftrain.drop(['Actual'], axis=1))
    temp.reset_index(drop=True,inplace=True)
    trainx = temp.values
    mlp = LinearRegression()
    mlp.fit(trainx,trainy)
    
    return(mlp)
  

def rebound_input(df_this_season, oppTeam, ownTeam, day, player, place):
    #opposing team dataframes and dates
    df_Opp_Ag=df_this_season[df_this_season['OPPONENT \nTEAM']==oppTeam]
    df_Opp_For=df_this_season[df_this_season['OWN \nTEAM']==oppTeam]
    oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
    #own team dataframes and dates
    df_Own_Ag = df_this_season[df_this_season['OPPONENT \nTEAM']==ownTeam]
    df_Own_For = df_this_season[df_this_season['OWN \nTEAM']==ownTeam]
    ownDates=(df_Own_For[df_Own_For['DATE']<day])['DATE'].unique()
    #player dataframes
    df_Player=df_this_season[df_this_season['PLAYER \nFULL NAME']==player]
    actual=df_Player[df_Player['DATE']==day]['TOT'].iloc[0]
    df_Player=df_Player[df_Player['DATE']<day]

    #now use 1 'if' to check if all 3 pieces from above are valid
    if (len(oppDates)-N>=0) & (len(ownDates)-N>=0) & (df_Player.shape[0]>=N):
#    oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
#    if len(oppDates)-N>=0:
        dayNbackOpp=oppDates[len(oppDates)-N]
        #Only want data for the opposing teams previous N games
        df_Opp_Ag=df_Opp_Ag[(df_Opp_Ag['DATE']<day) & (df_Opp_Ag['DATE']>=dayNbackOpp)]
        df_Opp_For=df_Opp_For[(df_Opp_For['DATE']<day) & (df_Opp_For['DATE']>=dayNbackOpp)]
        #Need to fix the criteria here once date problem is fixed
        
    	  #Calculate metrics for opposing team
        Opp_FGperc_Ag=df_Opp_Ag['FG'].sum()/df_Opp_Ag['FGA'].sum()
        Opp_FGperc_For=df_Opp_For['FG'].sum()/df_Opp_For['FGA'].sum()
        Opp_FG_Ag=df_Opp_Ag['FG'].sum()
        Opp_FG_For=df_Opp_For['FG'].sum()
        Opp_REB_For=df_Opp_For['TOT'].sum()
        Opp_REB_Ag=df_Opp_Ag['TOT'].sum()
        Opp_OREB_Ag=df_Opp_Ag['OR'].sum()
        Opp_DREB_Ag=df_Opp_Ag['DR'].sum()
        
        #Calculate metrics for own team 
#        df_Own_Ag = df_this_season[df_this_season['OPPONENT \nTEAM']==ownTeam]
#        df_Own_For = df_this_season[df_this_season['OWN \nTEAM']==ownTeam]
#        ownDates=(df_Own_For[df_Own_For['DATE']<day])['DATE'].unique()
#        if len(ownDates)-N>=0:
        dayNbackOwn=ownDates[len(ownDates)-N]
        #Only want data for the own teams previous N games
        df_Own_Ag=df_Own_Ag[(df_Own_Ag['DATE']<day) & (df_Own_Ag['DATE']>=dayNbackOwn)]
        df_Own_For=df_Own_For[(df_Own_For['DATE']<day) & (df_Own_For['DATE']>=dayNbackOwn)]
        #Calculate metrics for own team
        Own_FGperc_Ag=df_Own_Ag['FG'].sum()/df_Own_Ag['FGA'].sum()
        Own_FGperc_For=df_Own_For['FG'].sum()/df_Own_For['FGA'].sum()
        Own_FG_Ag=df_Own_Ag['FG'].sum()
        Own_FG_For=df_Own_For['FG'].sum()
            
            #metrics for the player
#            df_Player=df_this_season[df_this_season['PLAYER \nFULL NAME']==player]
#            actual=df_Player[df_Player['DATE']==day]['TOT'].iloc[0]
#            df_Player=df_Player[df_Player['DATE']<day]
#            if df_Player.shape[0]>=N:
        Player_Rebounds=((df_Player['TOT'].tail(N)).sum())/N
        Player_Rebounds_Short=((df_Player['TOT'].tail(N2)).sum())/N2
        DaysOff=day-df_Player['DATE'].iloc[-1]
        Player_ORebounds=df_Player['OR'].tail(N).sum()
        Player_DRebounds=df_Player['DR'].tail(N).sum()
        Median_Rebounds=df_Player['TOT'].tail(N).median()
        Std_Rebounds=df_Player['TOT'].tail(N).std()
        Max_Rebounds=df_Player['TOT'].tail(N).max()-((df_Player['TOT'].tail(N)).sum())/N
        Min_Rebounds=df_Player['TOT'].tail(N).min()-((df_Player['TOT'].tail(N)).sum())/N

#        predictionInput = [Player_Rebounds,Player_Rebounds_Short,
#                                      Median_Rebounds, Max_Rebounds, Min_Rebounds,
#                                      Player_ORebounds, Player_DRebounds,
#                                      Opp_OREB_Ag, Opp_DREB_Ag,
#                                      Own_FGperc_Ag,Own_FGperc_For,Own_FG_Ag,Own_FG_For,
#                                      Opp_FGperc_Ag,Opp_FGperc_For,Opp_FG_Ag,Opp_FG_For,
#                                      Opp_REB_For,Opp_REB_Ag,
#                                      DaysOff.days,place]
        predictionInput = [Player_Rebounds, Own_FGperc_Ag, Own_FGperc_For,
         Opp_FGperc_Ag, Opp_FGperc_For, place]
    else:
        predictionInput = [np.nan]
        
    return(predictionInput)


df_bets=pd.DataFrame()
#Now for each potential bet
#for i in range(df_odds.shape[0]):
for i in range(500, 1500):
    df_bet=df_odds.ix[i]
    mid = df_bet['#']
    o=df_bet['over_price']
    u=df_bet['under_price']
    overIP=1/o
    underIP=1/u
    player=df_bet['New Name']
    day=df_bet['dateGame']
    actual = df_bet['Actual']
    df_game=df_this_season[(df_this_season['PLAYER \nFULL NAME']==player)&(df_this_season['DATE']==day)]
    oppTeam=df_game['OPPONENT \nTEAM'].iloc[0] 
    ownTeam=df_game['OWN \nTEAM'].iloc[0] 
    place=df_game['VENUE\n(R/H)'].iloc[0]
    if place=='R':
        place=0
    else:
        place=1

    predictionInput = rebound_input(df_this_season, oppTeam, ownTeam, day, player, place)
    
    if not(np.isnan(predictionInput[0])):             
        
        predictionInput = np.array(predictionInput).reshape(1,-1)
        
        model = model_selection(mid)
        yHat= model.predict_proba(predictionInput)
        
        #just see what they regressor model predicts
        regressor = model_selection_regressor(mid).predict(predictionInput)
        
        
        overPredictedIP=yHat[0,1]
        underPredictedIP=yHat[0,0]
    
        
        returnO=yHat[0,1]*(o-1)-yHat[0,0]
        returnU=-yHat[0,1]+yHat[0,0]*(u-1)
        
        if (returnU > 0):
            under = True
            over = False
        elif (returnO > 0):
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
#        df_temp=pd.DataFrame([player, day,over, under,o,u, mid,actual,overIP,overPredictedIP,underIP,underPredictedIP,returnO, returnU, win, money]).transpose()
        temp=[player, day, u, returnU, returnO, o, mid,
              underPredictedIP, overPredictedIP, yHat, actual ,
              regressor,
              over, under,win,money]
        df_temp=pd.DataFrame(temp)
        df_temp=df_temp.T
        
        df_bets=pd.concat([df_bets, df_temp], axis=0)

    
            
#df_bets.columns=['player', 'day','over','under','o','u', 'mid','actual','overIP','overPredictedIP','underIP','underPredictedIP','returnO', 'returnU', 'win', 'money']    
df_bets.columns=['player', 'day', 'u', 'returnu', 'returnO', 'o', 'mid',
              'underPredictedIP', 'overPredictedIP', 'yHat', 'actual',
              'regressor',
              'over', 'under', 'win', 'money']
df_analysis = df_bets.loc[df_bets['money'] != 0]
print(np.sum(df_bets['money']))

print(df_bets['win'].sum())
print(df_analysis.shape[0])
var = df_analysis['money'].cumsum()