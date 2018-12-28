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
#df = df.drop(['player', 'day'], axis=1)
df = df.drop(['player', 'day',historical,'Player_ORebounds','Player_DRebounds'], axis=1)
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)

temp=(dftrain.drop(['Actual'], axis=1))
means=temp.mean()
stds=temp.std()

#################Format data for training and teting model#####################
#Training data
trainy = pd.to_numeric(dftrain['Actual'])
trainx = ((dftrain.drop(['Actual'], axis=1)-means)/stds).values
#Test data
testy = pd.to_numeric(dftest['Actual'])
testx = ((dftest.drop(['Actual'], axis=1)-means)/stds).values

######################Try a simple linear regression########################
model = LassoCV(cv=5, random_state=0,verbose=2).fit(trainx, trainy)
error1 = sk.metrics.mean_squared_error(trainy, model.predict(trainx))
error2 = sk.metrics.mean_squared_error(testy, model.predict(testx))
print("Linear regression results")
print("Mean Squared Error: train then test")
print(error1)
print(error2)
print(model.coef_)


#########################Optimize Boosted Tree#################################
#Optimize some parameters
param_test1 = {'hidden_layer_sizes':[(4,2),(8,4,2),(16,8,4,2),(32,16,8,4,2),(64,32,16,8,4,2),(128,64,32,16,8,4,2),(25,5)],
                                     'max_iter':[100,500,1000],
                                     'alpha':[0.001,0.01,0.1,1]}

gsearch1 = GridSearchCV(estimator = MLPRegressor(), iid=False,param_grid = param_test1,
                        scoring='neg_mean_squared_error',cv=5, verbose=2)
gsearch1.fit(trainx, trainy)
opt_hidden_layer_sizes=gsearch1.best_params_['hidden_layer_sizes']
opt_max_iter=gsearch1.best_params_['max_iter']
opt_alpha=gsearch1.best_params_['alpha']
#gsearch1.grid_scores_

#Test out the optimized model
model = MLPRegressor(hidden_layer_sizes=opt_hidden_layer_sizes,max_iter=opt_max_iter,alpha=opt_alpha)
model.fit(trainx, trainy)
error3 = sk.metrics.mean_squared_error(trainy,model.predict(trainx))
error4 = sk.metrics.mean_squared_error(testy,model.predict(testx))
print("Optimized NN results")
print(error3)
print(error4)
#temp=dftest
#temp['predicted']=model.predict_proba(testx)[:,1]
#temp['predicted2']=model.predict(testx)


###########################See how a naive model does##########################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df=df[['Actual',historical]]
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)
trainy = pd.to_numeric(dftrain['Actual'])
trainx = (dftrain.drop(['Actual'], axis=1)).values
testy = pd.to_numeric(dftest['Actual'])
testx = dftest.drop(['Actual'], axis=1).values
model = LinearRegression()
model.fit(trainx, trainy)
error5 = sk.metrics.mean_squared_error(trainy,model.predict(trainx))
error6 = sk.metrics.mean_squared_error(testy,model.predict(testx))
print("Naive model results")
print(error5)
print(error6)
print(model.coef_)
#temp['predictedNaive']=model.predict(testx)



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







