import pandas as pd
import sklearn as sk
import numpy as np
import os 
import os.path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


#os.chdir('Rebounds DataFrames')
os.chdir('C:\\GitHub\\nba-python\\Rebounds\\Rebounds Dataframes')


############################User defined stuff###############################

N=10
stat='RBS'
historical='Player_Rebounds'


#Portion of data for train, test and validation
test_size = 0.175
#ptrain=0.7
#ptest=0.2
#pval=0.1


############################Import and sort data###############################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df = df.drop(['player', 'day'], axis=1)




#Split and sort into train, test and validate based on proportions defined by user
#num=df.shape[0]
#dftrain=df.iloc[0:round(num*ptrain)]
#dftest=df.iloc[round(num*ptrain)+1:round(num*(ptrain+ptest))]
#dfvalidate=df.iloc[round(num*(ptrain+ptest))+1:num]


dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)


#################Format data for training and teting model#####################
#Training data
trainy = pd.to_numeric(dftrain['Actual'])
trainx = (dftrain.drop(['Actual'], axis=1)).values
#Test data
testy = pd.to_numeric(dftest['Actual'])
testx = dftest.drop(['Actual'], axis=1).values
#Validation data
#validatey = pd.to_numeric(dfvalidate['Actual'])
#validatex = dfvalidate.drop(['Actual'], axis=1).values

######################Try a simple linear regression########################
model = LinearRegression()
model.fit(trainx, trainy)
#error1 = sk.metrics.log_loss(trainy,model.predict_proba(trainx))
#error2 = sk.metrics.log_loss(testy,model.predict_proba(testx))
error1 = sk.metrics.mean_squared_error(trainy, model.predict(trainx))
error2 = sk.metrics.mean_squared_error(testy, model.predict(testx))
print("Linear regression results")
print("Mean Squared Error: train then test")
print(error1)
print(error2)
print(model.coef_)


#########################Optimize Boosted Tree#################################
#Initial guess for parameters
opt_max_depth=2
opt_min_split=5
opt_min_samples_leaf=5
opt_n_estimators=500
opt_learning_rate=0.1

#Optimize some parameters
param_test1 = {'n_estimators':[100,200,400,800,1600,3200],'learning_rate':[0.001,0.003,0.01,0.03,0.1],'max_depth':[1,2,3,4,5]}

gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(min_samples_split=opt_min_split,min_samples_leaf=opt_min_samples_leaf), iid=False, 
                                                              param_grid = param_test1,scoring='neg_mean_squared_error',
                                                              cv=5, verbose=0)
gsearch1.fit(trainx, trainy)
opt_n_estimators=gsearch1.best_params_['n_estimators']
opt_learning_rate=gsearch1.best_params_['learning_rate']
opt_max_depth=gsearch1.best_params_['max_depth']
#gsearch1.grid_scores_

#Optimize some parameters
param_test2 = {'min_samples_leaf':[10,25,50,100], 'min_samples_split':[10,25,50,100]}

gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=opt_n_estimators,max_depth=opt_max_depth,
                                                              learning_rate=opt_learning_rate),iid=False,
                                                              param_grid = param_test2,scoring='neg_mean_squared_error',
                                                              cv=5, verbose=0)
gsearch2.fit(trainx, trainy)
opt_min_samples_leaf=gsearch2.best_params_['min_samples_leaf']
opt_min_split=gsearch2.best_params_['min_samples_split']

#Test out the optimized model
model = GradientBoostingRegressor(n_estimators=opt_n_estimators,min_samples_split=opt_min_split,
                                max_depth=opt_max_depth,learning_rate=opt_learning_rate,
                                min_samples_leaf=opt_min_samples_leaf)
model.fit(trainx, trainy)

error3 = sk.metrics.mean_squared_error(trainy,model.predict(trainx))
error4 = sk.metrics.mean_squared_error(testy,model.predict(testx))
print("Optimized tree results")
print(error3)
print(error4)
features=model.feature_importances_
temp=dftest
temp = temp[['Actual']]
temp['predicted']=model.predict(testx)


###########################See how a naive model does##########################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df=df[['Actual',historical]]

#num=df.shape[0]
#dftrain=df.iloc[0:round(num*ptrain)]
#dftest=df.iloc[round(num*ptrain)+1:round(num*(ptrain+ptest))]
#dfvalidate=df.iloc[round(num*(ptrain+ptest))+1:num]

dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)
trainy = pd.to_numeric(dftrain['Actual'])
trainx = (dftrain.drop(['Actual'], axis=1)).values
testy = pd.to_numeric(dftest['Actual'])
testx = dftest.drop(['Actual'], axis=1).values

#validatey = pd.to_numeric(dfvalidate['Actual'])
#validatex = dfvalidate.drop(['Actual'], axis=1).values

model = LogisticRegression()
model.fit(trainx, trainy)
error5 = sk.metrics.mean_squared_error(trainy,model.predict(trainx))
error6 = sk.metrics.mean_squared_error(testy,model.predict(testx))
print("Naive model results")
print(error5)
print(error6)
print(model.coef_)
temp['predictedNaive']=model.predict(testx)



###########################Output the results##########################


df_out=pd.DataFrame([N,dataFile,opt_n_estimators,opt_min_split,opt_max_depth,
        opt_learning_rate,opt_min_samples_leaf,error1,error2,error3,
        error4,error5,error6,test_size]).transpose()

df_out.columns=['N', 'dataFile','opt_n_estimators','opt_min_split','opt_max_depth',
                'opt_learning_rate','opt_min_samples_leaf','Linear train','Linear test',
                'Boost train','Boost test','Naive train','Naive test','test size']

if os.path.isfile(stat+' results.csv'):
    df_out_old = pd.read_csv(stat+' results.csv')
    df_out = pd.concat([df_out_old, df_out])


df_out.to_csv(stat+' results.csv',index=False)







