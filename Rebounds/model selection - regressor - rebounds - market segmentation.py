import pandas as pd
import sklearn as sk
import numpy as np
import os 
import os.path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


os.chdir('C:\\GitHub\\nba-python\\Rebounds\\Rebounds Dataframes')


############################User defined stuff#################################
N=10
stat='RBS'
historical='Player_Rebounds'

#Portion of data for train, test and validation
test_size = 0.25

############################Import and sort data###############################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df = df.drop(['player', 'day'], axis=1)
df = df[['Actual', 'Player_Rebounds']]
dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)


#################Format data for training and testing##########################
#Training data
trainy = pd.to_numeric(dftrain['Actual'])
trainx = (dftrain.drop(['Actual'], axis=1)).values
#trainx = dftrain[['Player_Rebounds']].values
#Test data
testy = pd.to_numeric(dftest['Actual'])
testx = dftest.drop(['Actual'], axis=1).values
#testx = dftest[['Player_Rebounds']].values



###########Find different market segments based on main indicators#############

kmeans = KMeans(n_clusters=3, random_state=101)
kmeans.fit(trainx)
dftrain['cluster'] = kmeans.labels_
test_cluster_labels = kmeans.predict(testx)
dftest['cluster'] = test_cluster_labels

dftrain1 = dftrain.loc[dftrain['cluster'] == 1]
dftrain1 = dftrain1.drop(['cluster'], axis=1)
trainx1 = dftrain1.drop(['Actual'], axis=1).values
trainy1 = pd.to_numeric(dftrain1['Actual'])

dftest1 = dftest.loc[dftest['cluster'] == 1]
dftest1 = dftest1.drop(['cluster'], axis=1)
testx1 = dftest1.drop(['Actual'], axis=1).values
testy1 = pd.to_numeric(dftest1['Actual'])

model1 = LinearRegression(fit_intercept=True)
model1.fit(trainx1, trainy1)
error1 = sk.metrics.mean_squared_error(trainy1, model1.predict(trainx1))
error2 = sk.metrics.mean_squared_error(testy1, model1.predict(testx1))
print("Linear regression results")
print("Mean Squared Error: train then test")
print(error1)
print(error2)
#print(model1.coef_)
labels = dftrain.drop(['Actual', 'cluster'], axis=1).columns
coefficients = model1.coef_
df_coefficients = pd.DataFrame(data=[labels,coefficients])

#model1_tree = MLPRegressor()
#model1_tree.fit(trainx1, trainy1)
#error1 = sk.metrics.mean_squared_error(trainy1, model1_tree.predict(trainx1))
#error2 = sk.metrics.mean_squared_error(testy1, model1_tree.predict(testx1))
#print("Gradient Boosting Regressor results")
#print("Mean Squared Error: train then test")
#print(error1)
#print(error2)











#
#######################Try a simple linear regression###########################
#model = LinearRegression()
#model.fit(trainx, trainy)
#error1 = sk.metrics.mean_squared_error(trainy, model.predict(trainx))
#error2 = sk.metrics.mean_squared_error(testy, model.predict(testx))
#print("Linear regression results")
#print("Mean Squared Error: train then test")
#print(error1)
#print(error2)
#print(model.coef_)
#
#
############################See how a naive model does##########################
#dataFile=stat+' dataframe N'+str(N)+'.pkl'
#df = pd.read_pickle(dataFile)
#df = df.dropna()
#df=df[['Actual',historical]]
#
#
#dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)
#trainy = pd.to_numeric(dftrain['Actual'])
#trainx = (dftrain.drop(['Actual'], axis=1)).values
#testy = pd.to_numeric(dftest['Actual'])
#testx = dftest.drop(['Actual'], axis=1).values
#
#
#
#model = LinearRegression()
#model.fit(trainx, trainy)
#error5 = sk.metrics.mean_squared_error(trainy,model.predict(trainx))
#error6 = sk.metrics.mean_squared_error(testy,model.predict(testx))
#print("Naive model results")
#print(error5)
#print(error6)
#print(model.coef_)
##temp['predictedNaive']=model.predict(testx)
#
#
#
############################Output the results##########################
#
#"""
#df_out=pd.DataFrame([N,dataFile,opt_n_estimators,opt_min_split,opt_max_depth,
#        opt_learning_rate,opt_min_samples_leaf,error1,error2,error3,
#        error4,error5,error6,test_size]).transpose()
#
#df_out.columns=['N', 'dataFile','opt_n_estimators','opt_min_split','opt_max_depth',
#                'opt_learning_rate','opt_min_samples_leaf','Linear train','Linear test',
#                'Boost train','Boost test','Naive train','Naive test','test size']
#
#if os.path.isfile(stat+' results.csv'):
#    df_out_old = pd.read_csv(stat+' results.csv')
#    df_out = pd.concat([df_out_old, df_out])
#
#
#df_out.to_csv(stat+' results.csv',index=False)
#"""
#





