import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, TensorDataset, DataLoader

os.chdir('C:\\GitHub\\nba-python\\Fantasy')
#load to dfs data
df_dfs = pd.read_excel('01-01-2020-nba-season-dfs-feed.xlsx', header=1)
df_dfs.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER', 'OWN TEAM',
       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'DAYS REST', 
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO',
       'SAL - DRAFTKINGS','SAL - FANDUEL', 'SAL - YAHOO',
       'FPS - DRAFTKINGS', 'FPS - FANDUEL', 'FPS - YAHOO']
df_dfs.drop(columns= ['DATASET', 'GAME-ID', 'PLAYER-ID', 'OWN TEAM',
       'OPPONENT TEAM', 'STARTER (Y/N)', 'VENUE (R/H)', 'MINUTES',
       'USAGE RATE', 'DAYS REST', 
       'POS - DRAFTKINGS', 'POS - FANDUEL', 'POS - YAHOO', 
       'SAL - YAHOO',
       'FPS - YAHOO'], inplace=True)
#load the player data
df_player = pd.read_excel('01-01-2020-nba-season-player-feed.xlsx')
df_player.columns = ['DATASET', 'GAME-ID', 'DATE', 'PLAYER-ID', 'PLAYER',
       'POSITION', 'OWN TEAM', 'OPPONENT TEAM', 'VENUE (R/H)',
       'STARTER (Y/N)', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR',
       'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'USAGE RATE',
       'DAYS REST']
df_player.drop(columns = ['GAME-ID', 'PLAYER-ID', 'STARTER (Y/N)', 'USAGE RATE', 'DAYS REST'], inplace=True)
#merge dfs and player
df_merged = df_player.merge(df_dfs, how='left', left_on=['DATE', 'PLAYER'],
                            right_on=['DATE', 'PLAYER'])
df_merged['DATE'] = pd.to_datetime(df_merged['DATE'])


del df_dfs, df_player
#inputs: player, date, player's team, opponents's team, player home/away
#use Draftking data to get these inputs
os.chdir('C:\\GitHub\\nba-python\\Fantasy')
df_dk = pd.read_csv('DKSalaries (10).csv', skiprows=6)
df_dk.drop(columns=['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4',
       'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8','Position',
       'Name + ID','ID', 'Roster Position', 'Salary',
       'AvgPointsPerGame'], inplace=True)

df_dk['Team1'] = df_dk['Game Info'].str.split("@", n = 1, expand = True)[0]
df_dk['Team2'] = df_dk['Game Info'].str.split("@", n = 1, expand = True)[1].str.split(" ", n = 1, expand = True)[0]
df_dk['Date'] = df_dk['Game Info'].str.split("@", n = 1, expand = True)[1].str.split(" ", n = 1, expand = True)[1]
df_dk['Date'] = df_dk['Date'].str.split(" ", n = 1, expand = True)[0]
df_dk['Date'] = pd.to_datetime(df_dk['Date'])
df_dk['Player Team'] = ''
df_dk['Opp Team'] = ''
df_dk['Home'] = 0
for i in range(df_dk.shape[0]):
    team = df_dk['TeamAbbrev'].iloc[i]
    team1 = df_dk['Team1'].iloc[i]
    team2 = df_dk['Team2'].iloc[i]
    if team == team1:
        df_dk['Player Team'].iloc[i] = team1
        df_dk['Opp Team'].iloc[i] = team2
    elif team == team2:
        df_dk['Player Team'].iloc[i] = team2
        df_dk['Opp Team'].iloc[i] = team1
        df_dk['Home'].iloc[i] = 1
#df_dk.drop(columns=['TeamAbbrev','Team1','Team2'], inplace=True)
        
#replace team name abbreviations with full names
os.chdir('C:\\GitHub\\nba-python\\BigDataBall Data')
df_ref = pd.read_excel('Team Abbreviations.xlsx')
df_ref['INITIALS'] = df_ref['INITIALS'].str.upper()
df_ref.set_index(df_ref['INITIALS'], inplace=True)
df_ref.drop(columns=['INITIALS'], inplace=True)
team_names = df_ref.to_dict()
team_names = team_names['SHORT NAME']
df_dk.replace(team_names, inplace=True)

player = df_dk['Name'].iloc[0]
day= df_dk['Date'].iloc[0]
ownTeam= df_dk['Player Team'].iloc[0]
oppTeam = df_dk['Opp Team'].iloc[0]
place = df_dk['Home'].iloc[0]
df = df_merged.copy()
df.rename(columns={'FPS - DRAFTKINGS':'FPS','OPPONENT TEAM':'OPP TEAM', 'PLAYER':'PLAYER FULL NAME'},
          inplace=True)
N=5
N2=2


def partTwo(df, player, day, ownTeam, oppTeam, place):
    
    threshold=3
#    df_Game = df.loc[df['GAME-ID'] == GameID]
#    oppTeam=df_Game['OPP TEAM'].iloc[0]
#    ownTeam=df_Game['OWN TEAM'].iloc[0]
#    day=df_Game['DATE'].iloc[0]
#    place=df_Game['VENUE (R/H)'].iloc[0]
    df_out=pd.DataFrame()        

    
	 ################################################################################
	 #Opposing team data
     #games where the opposing team was the opposing team
    df_Opp_Ag=df[df['OPP TEAM']==oppTeam]
    #games where the opposing team was the subject team
    df_Opp_For=df[df['OWN TEAM']==oppTeam]
    
	 #What was the date of the opposing teams last N
    oppDates=(df_Opp_Ag[df_Opp_Ag['DATE']<day])['DATE'].unique()
    if len(oppDates)-N>=0:
        dayNback=oppDates[len(oppDates)-N]
    
    	  #Only want data for the opposing teams previous N games
        df_Opp_Ag=df_Opp_Ag[(df_Opp_Ag['DATE']<day) & (df_Opp_Ag['DATE']>=dayNback)]
        df_Opp_For=df_Opp_For[(df_Opp_For['DATE']<day) & (df_Opp_For['DATE']>=dayNback)]

    	#Calculate metrics for opposing team
        Opp_FPS_Ag=df_Opp_Ag['FPS'].sum()
        Opp_FPS_For=df_Opp_For['FPS'].sum()
        

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
            Own_FPS_Ag=df_Own_Ag['FPS'].sum()
            Own_FPS_For=df_Own_For['FPS'].sum()
            
            


            ##############################################################################
        	   #Go through each player
#            for i in df_Game['PLAYER FULL NAME']:
            df_Player=df[df['PLAYER FULL NAME']==player]
#            actual=df_Player[df_Player['DATE']==day]['FPS'].iloc[0] 
            #Only want data before today to calculate historical statistics
            df_Player=df_Player[df_Player['DATE']<day]
            if df_Player.shape[0]>=N:
                if sum(df_Player['FPS'].tail(N))>threshold*N:
                    #Calculate player statistics
                    Player_FPS=((df_Player['FPS'].tail(N)).sum())/N
                    Player_FPS_Short=((df_Player['FPS'].tail(N2)).sum())/N2
                    DaysOff=day-df_Player['DATE'].iloc[-1]
                    Median_FPS=df_Player['FPS'].tail(N).median()
                    Std_FPS=df_Player['FPS'].tail(N).std()
                    Max_FPS=df_Player['FPS'].tail(N).max()-((df_Player['FPS'].tail(N)).sum())/N
                    Min_FPS=df_Player['FPS'].tail(N).min()-((df_Player['FPS'].tail(N)).sum())/N
                    Player_Points = df_Player['PTS'].tail(N).sum()/N
                    Player_TOT = df_Player['TOT'].tail(N).sum()/N
                    Player_Minutes = df_Player['MIN'].tail(N).sum()/N
                    Player_FG = df_Player['FG'].tail(N).sum()/N
                    Player_FGA = df_Player['FGA'].tail(N).sum()/N
                    Player_3P = df_Player['3P'].tail(N).sum()/N
                    Player_3PA = df_Player['3PA'].tail(N).sum()/N
                    Player_Assists = df_Player['A'].tail(N).sum()/N
                    Player_Fouls = df_Player['PF'].tail(N).sum()/N
                    Player_Turnovers = df_Player['TO'].tail(N).sum()/N
                    Player_Blocks = df_Player['BL'].tail(N).sum()/N
                    #think about putting salary, prob not though since will heavily influence model with their predictions
                    #Put result together with indicators
                    global var_list

                    var_list = {'Player_FPS':Player_FPS, 'Player_FPS_Short':Player_FPS_Short,
                                          'Median_FPS':Median_FPS, 'Std_FPS':Std_FPS, 'Max_FPS':Max_FPS,
                                          'Min_FPS':Min_FPS, 'Player_Points':Player_Points, 'Player_TOT':Player_TOT,
                                          'Player_Minutes':Player_Minutes, 
                                          'Player_FG':Player_FG, 'Player_FGA':Player_FGA,
                                          'Player_3P':Player_3P, 'Player_3PA':Player_3PA,
                                          'Player_Assists':Player_Assists, 'Player_Fouls':Player_Fouls,
                                          'Player_Turnovers':Player_Turnovers, 'Player_Blocks':Player_Blocks,
                                          'Opp_FPS_Ag':Opp_FPS_Ag, 'Opp_FPS_For':Opp_FPS_For,
                                          'Own_FPS_Ag':Own_FPS_Ag, 'Own_FPS_For':Own_FPS_For,
                                          'Own_Team':ownTeam, 'Opp_team':oppTeam,
                                          'DaysOff':DaysOff,'place':place,'Player':player,'day':day}
                    df_temp=pd.DataFrame.from_dict(var_list, orient='index').transpose()                       
                    df_out=pd.concat([df_out, df_temp])
    return df_out
        
sample = partTwo(df, player, day, ownTeam, oppTeam, place)


#predict using the trained points
class NeuralNet(nn.Module):
    def __init__(self, input_size, layers_size, num_layers, output_size):
        super(NeuralNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
        for i in range(1, num_layers - 1):
            self.linears.extend([nn.Linear(layers_size, layers_size)])
        self.linears.append(nn.Linear(layers_size, output_size))
        self.sigmoid = nn.Sigmoid()

        
    
    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
        x = self.linears[i+1](x)
        return x

input_size = 23
output_size = 1
layers_size = 5
num_layers = 5

model = NeuralNet(input_size, layers_size, num_layers, output_size)

os.chdir('C:\\GitHub\\nba-python\\Fantasy\\Trained Models')
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

