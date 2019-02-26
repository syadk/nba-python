import pandas as pd
import sklearn as sk
import numpy as np
import os 
import os.path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE



os.chdir('C:\\GitHub\\nba-python\\Rebounds\\Rebounds Dataframes')


############################User defined stuff#################################
N=5
stat='RBS'
historical='Player_Rebounds'

#Portion of data for train, test and validation
test_size = 0.25


cluster = 2

############################Import and sort data###############################
dataFile=stat+' dataframe N'+str(N)+'.pkl'
df = pd.read_pickle(dataFile)
df = df.dropna()
df = df.drop(['player', 'day'], axis=1)

#df = df[['Actual', 'Player_Rebounds', 'Opp_REB_Ag', 'Opp_FGA_For', 'place']]
#df = df[['Actual', 'Player_Rebounds',
#         
#         'Opp_FGperc_For',
#         'Opp_FGA_For'
#         ]]

#df = df[['Actual', 'Player_Rebounds']]

dftrain, dftest = train_test_split(df, test_size=test_size, random_state=101)



#################Format data for training and testing##########################
#Training data
trainy = pd.to_numeric(dftrain['Actual'])
#trainx = (dftrain.drop(['Actual'], axis=1)).values
trainx = dftrain[['Player_Rebounds']]

#Test data
testy = pd.to_numeric(dftest['Actual'])
#testx = dftest.drop(['Actual'], axis=1).values
testx = dftest[['Player_Rebounds']]


###########Find different market segments based on main indicators#############

kmeans = KMeans(n_clusters=5, random_state=101)
kmeans.fit(trainx)
dftrain['cluster'] = kmeans.labels_
test_cluster_labels = kmeans.predict(testx)
dftest['cluster'] = test_cluster_labels

########### new insert just for testing, remove soon
#dftrain = dftrain.drop(['Player_Rebounds'], axis=1)
#dftest = dftest.drop(['Player_Rebounds'], axis=1)
#dftrain = dftrain[['Actual', 'Player_Rebounds', 'Opp_REB_Ag', 'Opp_FGA_For', 'place', 'cluster']]
#dftest = dftest[['Actual', 'Player_Rebounds', 'Opp_REB_Ag', 'Opp_FGA_For', 'place', 'cluster']]



dftrain1 = dftrain.loc[dftrain['cluster'] == cluster]
dftrain1 = dftrain1.drop(['cluster'], axis=1)
dftrain1.reset_index(inplace=True, drop=True)
trainx1 = dftrain1.drop(['Actual'], axis=1)
trainy1 = pd.to_numeric(dftrain1['Actual'])

dftest1 = dftest.loc[dftest['cluster'] == cluster]
dftest1 = dftest1.drop(['cluster'], axis=1)
dftest1.reset_index(inplace=True, drop=True)
testx1 = dftest1.drop(['Actual'], axis=1)
testy1 = pd.to_numeric(dftest1['Actual'])

select = SelectPercentile(percentile=20)
select.fit(trainx1, trainy1)
# transform training set
trainx1_selected = select.transform(trainx1)
mask = select.get_support()
#


model1 = LinearRegression()
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
df_coefficients = pd.DataFrame(data=[labels,coefficients,mask]).transpose()

df_analysis = pd.DataFrame(data = [testy1, model1.predict(testx1)]).transpose()

####### Classifiers ##########################################################
#i think a lower 'log loss' error is better



dftrain_c = dftrain.loc[dftrain['cluster'] == cluster]
dftrain_c = dftrain_c.drop(['cluster'], axis=1)
dftrain_c.loc[dftrain_c['Actual'] < kmeans.cluster_centers_[cluster][0], ['Actual']] = 0
dftrain_c.loc[dftrain_c['Actual'] > kmeans.cluster_centers_[cluster][0], ['Actual']] = 1
dftrain_c.reset_index(inplace=True, drop=True)
trainx_c = dftrain_c.drop(['Actual'], axis=1)
trainy_c = pd.to_numeric(dftrain_c['Actual'])


dftest_c = dftest.loc[dftest['cluster'] == cluster]
dftest_c = dftest_c.drop(['cluster'], axis=1)
dftest_c.loc[dftest_c['Actual'] < kmeans.cluster_centers_[cluster][0], ['Actual']] = 0
dftest_c.loc[dftest_c['Actual'] > kmeans.cluster_centers_[cluster][0], ['Actual']] = 1
dftest_c.reset_index(inplace=True, drop=True)
testx_c = dftest_c.drop(['Actual'], axis=1)
testy_c = pd.to_numeric(dftest_c['Actual'])

model_classifier = GradientBoostingClassifier(n_estimators=100,min_samples_split=10,
                                max_depth=1,learning_rate=0.1,
                                min_samples_leaf=10)
model_classifier.fit(trainx_c, trainy_c)
error3 = sk.metrics.log_loss(trainy_c, model_classifier.predict_proba(trainx_c))
error4 = sk.metrics.log_loss(testy_c, model_classifier.predict_proba(testx_c))
print("Logistic regression results")
print("log loss: train then test")
print(error3)
print(error4)

coefficients_classifier = model_classifier.feature_importances_
df_coefficients_classifier = pd.DataFrame(data=[labels,coefficients_classifier]).transpose()

df_analysis_classifier = pd.DataFrame(data = [testy1, model_classifier.predict(testx1)]).transpose()
######            now tree model ############################################
#opt_max_depth=2
#opt_min_split=5
#opt_min_samples_leaf=5
#opt_n_estimators=500
#opt_learning_rate=0.1
#
##Optimize some parameters
#param_test1 = {'n_estimators':[100,200,400,800,1600,3200],'learning_rate':[0.001,0.003,0.01,0.03,0.1],'max_depth':[1,2,3,4,5]}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_split=opt_min_split,min_samples_leaf=opt_min_samples_leaf), iid=False, 
#                                                              param_grid = param_test1,scoring='neg_log_loss',
#                                                              cv=5, verbose=0)
#gsearch1.fit(trainx_c, trainy_c)
#opt_n_estimators=gsearch1.best_params_['n_estimators']
#opt_learning_rate=gsearch1.best_params_['learning_rate']
#opt_max_depth=gsearch1.best_params_['max_depth']
##gsearch1.grid_scores_
#
##Optimize some parameters
#param_test2 = {'min_samples_leaf':[10,25,50,100], 'min_samples_split':[10,25,50,100]}
#gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=opt_n_estimators,max_depth=opt_max_depth,
#                                                              learning_rate=opt_learning_rate),iid=False,
#                                                              param_grid = param_test2,scoring='neg_log_loss',
#                                                              cv=5, verbose=0)
#gsearch2.fit(trainx_c, trainy_c)
#opt_min_samples_leaf=gsearch2.best_params_['min_samples_leaf']
#opt_min_split=gsearch2.best_params_['min_samples_split']
#
##Test out the optimized model
#model = GradientBoostingClassifier(n_estimators=opt_n_estimators,min_samples_split=opt_min_split,
#                                max_depth=opt_max_depth,learning_rate=opt_learning_rate,
#                                min_samples_leaf=opt_min_samples_leaf)
#model.fit(trainx_c, trainy_c)
#error5 = sk.metrics.log_loss(trainy_c,model.predict_proba(trainx_c))
#error6 = sk.metrics.log_loss(testy_c,model.predict_proba(testx_c))
#print("Optimized tree results")
#print(error5)
#print(error6)
#features=model.feature_importances_
#temp=dftest
#temp['predicted']=model.predict(testx1)



#learning rate = 0.1
#max dapth=1
#min samples leaf = 10
#min split = 10
#n_estimators = 100
####################### select some indicators using new methods ################










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
##print(model.coef_)
##
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





