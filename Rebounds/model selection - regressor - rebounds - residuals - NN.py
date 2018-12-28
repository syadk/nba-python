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
import matplotlib.pyplot as plt

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
df=df[['Actual',historical,'Max_Rebounds','Min_Rebounds','Player_Rebounds_Short','place']]
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=1)

#Scale each predictor to 1 std dev and 0 mean
temp=(dftrain.drop(['Actual'], axis=1))
means=temp.mean()
stds=temp.std()




#################Format data for training and testing model#####################
#Training data
trainy = pd.to_numeric(dftrain['Actual'])
dftrain=((dftrain-means)/stds)
trainx1 = pd.to_numeric(dftrain[historical]).values.reshape(-1, 1)
trainx2 = dftrain.drop(['Actual'], axis=1).values

#Test data
testy = pd.to_numeric(dftest['Actual'])
dftest=((dftest-means)/stds)
testx1 = pd.to_numeric(dftest[historical]).values.reshape(-1, 1)
testx2 = dftest.drop(['Actual'], axis=1).values



######################First fit a simple linear model using the main predictor########################
model = LinearRegression().fit(trainx1, trainy)
estimate = model.predict(trainx1)
trainy2=trainy-estimate


###################Now use the rest of the data to fit a model to those predictors####################

param_test1 = {'hidden_layer_sizes':[(4,2),(8,4,2),(16,8,4,2),(32,16,8,4,2),(64,32,16,8,4,2),(3,3),(3,3,3),(3,3,3,3),(3,3,3,3,3),(3,3,3,3,3,3),(3,3,3,3,3,3,3),(3,3,3,3,3,3,3,3)],
                                     'max_iter':[500,1000,2000],
                                     'alpha':[0.01,0.1,1,10]}

gsearch1 = GridSearchCV(estimator = MLPRegressor(), iid=False,param_grid = param_test1,
                        scoring='neg_mean_squared_error',cv=4, verbose=2)
gsearch1.fit(trainx2, trainy2)
opt_hidden_layer_sizes=gsearch1.best_params_['hidden_layer_sizes']
opt_max_iter=gsearch1.best_params_['max_iter']
opt_alpha=gsearch1.best_params_['alpha']
#gsearch1.grid_scores_


model2 = MLPRegressor(hidden_layer_sizes=opt_hidden_layer_sizes,max_iter=opt_max_iter,alpha=opt_alpha)
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

#plt.scatter(testy,prediction)
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.xlim(0,20)
#plt.ylim(0,20)

plt.scatter(testy,prediction-testy)
plt.xlabel('Actual')
plt.ylabel('Residual')

###########################See how a naive model does##########################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df=df[['Actual',historical]]
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=1)
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

#plt.scatter(testy,model.predict(testx))
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.xlim(0,30)
#plt.ylim(0,30)

#plt.scatter(testy,model.predict(testx)-testy)
plt.scatter(model.predict(testx),model.predict(testx)-testy)
plt.xlabel('Actual')
plt.ylabel('Residual')
#plt.xlim(0,30)
#plt.ylim(0,30)

model.coef_
model.intercept_

temp=dftest
temp['predicted']=model.predict(testx)
temp['predicted2']=prediction


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






