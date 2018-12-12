import pandas as pd
import sklearn as sk
import numpy as np
import os 
import os.path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


os.chdir('C:\\GitHub\\nba-python\\Rebounds\\Rebounds Dataframes')


############################User defined stuff###############################
N=20
stat='RBS'
historical='Player_Rebounds'

#Portion of data for train and test
test_size = 0.25

############################Import and sort data###############################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df = df.drop(['player', 'day'], axis=1)
df=df[['Actual',historical,'Median_Rebounds','Max_Rebounds','Min_Rebounds','place']]
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=0)

#################Format data for training and testing model#####################
#Training data
trainy = pd.to_numeric(dftrain['Actual'])
trainx1 = pd.to_numeric(dftrain[historical]).values.reshape(-1, 1)
trainx2 = dftrain.drop(['Actual'], axis=1).values

#Test data
testy = pd.to_numeric(dftest['Actual'])
testx1 = pd.to_numeric(dftest[historical]).values.reshape(-1, 1)
testx2 = dftest.drop(['Actual'], axis=1).values



######################First fit a simple linear model using the main predictor########################
model = LinearRegression().fit(trainx1, trainy)
estimate = model.predict(trainx1)
trainy2=trainy-estimate


###################Now use the rest of the data to fit a model to those predictors####################
#Initial guess for parameters
opt_max_depth=2
opt_min_split=50
opt_min_samples_leaf=10
opt_n_estimators=500
opt_learning_rate=0.1

#Optimize some parameters
param_test1 = {'n_estimators':[100,250,500,1000],'learning_rate':[0.001,0.01,0.1,1],'max_depth':[1,2,3,5]}

gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(min_samples_split=opt_min_split,min_samples_leaf=opt_min_samples_leaf), iid=False, 
                                                              param_grid = param_test1,scoring='neg_mean_squared_error',
                                                              cv=5, verbose=2)
gsearch1.fit(trainx2, trainy2)
opt_n_estimators=gsearch1.best_params_['n_estimators']
opt_learning_rate=gsearch1.best_params_['learning_rate']
opt_max_depth=gsearch1.best_params_['max_depth']
#gsearch1.grid_scores_


model2 = GradientBoostingRegressor(n_estimators=opt_n_estimators,min_samples_split=opt_min_split,
                                max_depth=opt_max_depth,learning_rate=opt_learning_rate,
                                min_samples_leaf=opt_min_samples_leaf)




model2.fit(trainx2, trainy2)

print("Results")
print("Train:")
residEstimate=model2.predict(trainx2)
prediction = estimate+residEstimate
error1=sk.metrics.mean_squared_error(trainy,prediction)
print(error1)

print("Test:")
estimate = model.predict(testx1)
residEstimate=model2.predict(testx2)
prediction = estimate+residEstimate
error2=sk.metrics.mean_squared_error(testy,prediction)
print(error2)



###########################See how a naive model does##########################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df=df[['Actual',historical]]
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=0)
trainy = pd.to_numeric(dftrain['Actual'])
trainx = (dftrain.drop(['Actual'], axis=1)).values
testy = pd.to_numeric(dftest['Actual'])
testx = dftest.drop(['Actual'], axis=1).values
model = LinearRegression()
model.fit(trainx, trainy)
error5 = sk.metrics.mean_squared_error(trainy,model.predict(trainx))
error6 = sk.metrics.mean_squared_error(testy,model.predict(testx))
print("Naive model results")
print("Train:")
print(error5)
print("Test:")
print(error6)




############################Output the results##########################
#
#df_out=pd.DataFrame([N,dataFile,opt_hidden_layer_sizes,opt_max_iter,opt_alpha,error1,error2,error3,
#        error4,error5,error6]).transpose()
#
#df_out.columns=['N', 'dataFile','opt_hidden_layer_sizes','opt_max_iter','opt_alpha','Logistic train','Logistic test',
#                'Boost train','Boost test','Naive train','Naive test']
#
#if os.path.isfile(stat+' results NN.csv'):
#    df_out_old = pd.read_csv(stat+' results NN.csv')
#    df_out = pd.concat([df_out_old, df_out])
#
#
#df_out.to_csv(stat+' results NN.csv',index=False)






