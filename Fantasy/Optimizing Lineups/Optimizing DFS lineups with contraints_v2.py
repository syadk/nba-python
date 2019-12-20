import pandas as pd
import os
from itertools import permutations
from pulp import *
import re
from datetime import datetime, timedelta

os.chdir('C:\\GitHub\\nba-python\\Fantasy\\Optimizing Lineups\\Daily Salaries')
df = pd.read_csv('DKSalaries_dec20.csv')
df.drop(columns=['Name + ID','Game Info', 'TeamAbbrev'], inplace=True)

os.chdir('C:\\GitHub\\nba-python\\Fantasy\\Optimizing Lineups\\Daily Injuries')
df_injuries = pd.read_excel('nba-injury-report.xlsx')
df_injuries = df_injuries[['Player', 'Status']]

df = df.merge(df_injuries, how="left", left_on='Name', right_on='Player')

df['Status'] = df['Status'].astype(str)
df = df.loc[df['Status'] == 'nan']
df.drop(columns=['Player', 'Status'], inplace=True)


def string_match(cell, string):
    if string in cell:
        return 1
    else:
        return 0

df['PG'] = df['Roster Position'].apply(string_match, args=('PG', ))
df['SG'] = df['Roster Position'].apply(string_match, args=('SG', ))
df['SF'] = df['Roster Position'].apply(string_match, args=('SF', ))
df['PF'] = df['Roster Position'].apply(string_match, args=('PF', ))
df['C'] = df['Roster Position'].apply(string_match, args=('C', ))
df['G'] = df['Roster Position'].apply(string_match, args=('G', ))
df['F'] = df['Roster Position'].apply(string_match, args=('F', ))
df['UTIL'] = df['Roster Position'].apply(string_match, args=('UTIL', ))


availables = df.copy()
availables.columns = ['Position', 'Name', 'ID', 'Roster Position', 'Salary',
       'Points', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
availables.set_index('Name', inplace=True)

players = availables.to_dict(orient='index')


salaries = {}
points = {}
ids = {}
names = []

pgs = []
sgs = []
sfs = []
    
for key, value in players.items():
    name = key
    salary = value['Salary']
    point = value['Points']
    salaries[name] = salary
    points[name] = point
    names.append(name)
    
for i in range(0, df.shape[0]):
    row = df.iloc[i]
    if row['PG'] == 1:
        pgs.append(row['Name'])
    if row['SG'] == 1:
        sgs.append(row['Name'])
    if row['SF'] == 1:
        sfs.append(row['Name'])
        


pg = pulp.LpVariable.dicts("pg", names, cat="Binary")
sg = pulp.LpVariable.dicts("sg", names, cat="Binary")
sf = pulp.LpVariable.dicts("sf", names, cat="Binary")
pf = pulp.LpVariable.dicts("pf", names, cat="Binary")
c = pulp.LpVariable.dicts("c", names, cat="Binary")
f = pulp.LpVariable.dicts("f", names, cat="Binary")
g = pulp.LpVariable.dicts("g", names, cat="Binary")
util = pulp.LpVariable.dicts("util", names, cat="Binary")



prob = LpProblem("Fantasy", LpMaximize)
rewards = []
costs = []
position_constraints = []

#    
for name, point in points.items():
    costs += lpSum([salaries[name] * pg[name]])
    costs += lpSum([salaries[name] * sg[name]])
    costs += lpSum([salaries[name] * sf[name]])
    costs += lpSum([salaries[name] * pf[name]])
    costs += lpSum([salaries[name] * c[name]])
    costs += lpSum([salaries[name] * f[name]])
    costs += lpSum([salaries[name] * g[name]])
    costs += lpSum([salaries[name] * util[name]])

    rewards += lpSum([points[name] * pg[name]  * availables.loc[name, 'PG']]) # multiple by 3rd term so they can't get points for
    rewards += lpSum([points[name] * sg[name]  * availables.loc[name, 'SG']]) # positions they don't play for
    rewards += lpSum([points[name] * sf[name]  * availables.loc[name, 'SF']])
    rewards += lpSum([points[name] * pf[name]  * availables.loc[name, 'PF']])
    rewards += lpSum([points[name] * c[name]  * availables.loc[name, 'C']])
    rewards += lpSum([points[name] * f[name]  * availables.loc[name, 'F']])
    rewards += lpSum([points[name] * g[name]  * availables.loc[name, 'G']])
    rewards += lpSum([points[name] * util[name]  * availables.loc[name, 'UTIL']]) 

#no player should be selected twice
for name in pg:
    prob += lpSum([pg[name] + sg[name] + sf[name] + pf[name] + c[name] + f[name] + g[name] + util[name]]) <= 1


    
prob += lpSum(rewards)
prob += lpSum(costs) <= 50000

prob += lpSum(lpSum(pg) + lpSum(sg) + lpSum(sf) + lpSum(pf) + lpSum(c) + lpSum(f) +  lpSum(g) + lpSum(util)) <= 8 
prob += lpSum(pg) <= 1
prob += lpSum(sg) <= 1
prob += lpSum(sf) <= 1
prob += lpSum(pf) <= 1
prob += lpSum(c) <= 1
prob += lpSum(f) <= 1
prob += lpSum(g) <= 1
prob += lpSum(util) <= 1

#
#prob += lpSum(pg) >= 1
#prob += lpSum(sg) >= 1
#prob += lpSum(sf) >= 1
#prob += lpSum(pf) >= 1
#prob += lpSum(c) >= 1
#prob += lpSum(f) >= 1
#prob += lpSum(g) >= 1

prob.solve()

def summary(prob):
    div = '---------------------------------------\n'
    print("Variables:\n")
    score = str(prob.objective)
    constraints = [str(const) for const in prob.constraints.values()]
    for v in prob.variables():
        score = score.replace(v.name, str(v.varValue))
        constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
        if v.varValue != 0:
            print(v.name, "=", v.varValue)
    print(div)
    print("Constraints:")
    for constraint in constraints:
        constraint_pretty = " + ".join(re.findall("[0-9\.]*\*1.0", constraint))
        if constraint_pretty != "":
            print("{} = {}".format(constraint_pretty, eval(constraint_pretty)))
    print(div)
    print("Score:")
    score_pretty = " + ".join(re.findall("[0-9\.]+\*1.0", score))
    print("{} = {}".format(score_pretty, eval(score_pretty)))
summary(prob)
    
names = []
names_unclean = []
count = -1
for i in prob.variables():
    if i.varValue != 0:
        name = i.name
        name_1 = name.split("_",1)[1]
        name_2 = name_1.replace("_", " ")
        print(name_2)
        names.append(name_2)
        names_unclean.append(name)

dfview = pd.DataFrame()
for i in names:
    count = count + 1
    dftemp = df.loc[df['Name'] == i]
    dftemp['name_unclean'] = names_unclean[count]
    print(names_unclean[count])
    dfview = pd.concat([dfview, dftemp])

dfview['Optimized Position'] =  dfview['name_unclean'].str.split(pat="_")
dfview['Optimized Position']  = dfview['Optimized Position'].map(lambda x: x[0])
salary_used = dfview['Salary'].sum()
total_points = dfview['AvgPointsPerGame'].sum()
print(salary_used, total_points)    
#    
    
df_upload = pd.DataFrame(columns=['PG','SG','SF','PF','C','G','F','UTIL'], index=[1])
for i in range(0, dfview.shape[0]):
    v = dfview['Optimized Position'].iloc[i]
    id_number = dfview['ID'].iloc[i]
    if v == 'pg':
        pg = id_number
    elif v == 'sg':
        sg = id_number
    elif v == 'sf':
        sf = id_number
    elif v == 'pf':
        pf = id_number
    elif v == 'c':
        c = id_number
    elif v == 'g':
        g = id_number
    elif v == 'f':
        f = id_number
    elif v == 'util':
        util = id_number

df_upload['PG'].iloc[0] = pg
df_upload['SG'].iloc[0] = sg   
df_upload['SF'].iloc[0] = sf   
df_upload['PF'].iloc[0] = pf   
df_upload['C'].iloc[0] = c   
df_upload['G'].iloc[0] = g   
df_upload['F'].iloc[0] = f   
df_upload['UTIL'].iloc[0] = util   

today = str(datetime.today().day) + "_" + str(datetime.today().month)

os.chdir('C:\\GitHub\\nba-python\\Fantasy\\Optimizing Lineups\\Daily Lineup Uploads')
excelwriter = str("Lineup Upload_") + today + str(".xlsx")
df_upload.to_excel(excelwriter, index=False)