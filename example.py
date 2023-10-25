# IMPORTS ########################################################################################################
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
#opening data
import os
import pathlib
import warnings 
#used for plots
from scipy import stats
from mplsoccer import PyPizza, FontManager, add_image
# add image
from urllib.request import urlopen
from PIL import Image

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# GET DATA #######################################################################################################
path = os.path.join(str(pathlib.Path().resolve()), 'src', 'wyscout', 'events', 'events_England.json') 
with open(path) as f: 
    data = json.load(f) 
train = pd.DataFrame(data) 

train = train.loc[train.apply (lambda x: len(x.positions) == 2, axis = 1)]
##################################################################################################################

# TURNOEVERS #####################################################################################################
def turnovers(df):
    dribble_forward = df.loc[df["subEventName"] == "Ground attacking duel"]
    lost_dribble_forward = dribble_forward.loc[dribble_forward.apply(lambda x:{'id':701} in x.tags, axis = 1)]
    ldf = lost_dribble_forward.groupby(["playerId"]).eventId.count().reset_index()
    ldf.rename(columns = {'eventId':'lost_dribble_forward'}, inplace=True)
    
    passes = df.loc[df["eventName"] == "Pass"]
    lost_pass = passes.loc[passes.apply(lambda x:{'id':1802} in x.tags, axis = 1)]
    lp = lost_pass.groupby(["playerId"]).eventId.count().reset_index()
    lp.rename(columns = {'eventId':'lost_pass'}, inplace=True)

    turnovers_made = ldf.merge(lp, how = "outer", on = ["playerId"])

    return turnovers_made

turnover = turnovers(train)
##################################################################################################################

# CLEARANCES #####################################################################################################
def clearance(df):
    clearance = df.loc[df["subEventName"] == "Clearance"]
    clearing_player = clearance.groupby(["playerId"]).eventId.count().reset_index()
    clearing_player.rename(columns = {'eventId':'clearances'}, inplace=True)

    return clearing_player

clear = clearance(train)
##################################################################################################################

# INTERCEPTIONS ##################################################################################################
def interception(df):
    opp_pass = df.loc[df["eventName"] == "Pass"]
    interception = opp_pass.loc[opp_pass.apply(lambda x:{'id':1401} in x.tags, axis = 1)]
    intercepting_player = interception.groupby(["playerId"]).eventId.count().reset_index()
    intercepting_player.rename(columns = {'eventId':'interceptions'}, inplace=True)

    return intercepting_player

intercept = interception(train)
##################################################################################################################

# DUELS WON ###################################################################################################### 
def defensiveDuelsWon(df):
    loose_duels = df.loc[df["subEventName"] == "Ground loose ball duel"]
    won_loose_duels = loose_duels.loc[loose_duels.apply(lambda x:{'id':703} in x.tags, axis = 1)]
    wld_player = won_loose_duels.groupby(["playerId"]).eventId.count().reset_index()
    wld_player.rename(columns = {'eventId':'loose_duels_won'}, inplace=True)
    
    def_ground_duels = df.loc[df["subEventName"].isin(["Ground defending duel"])]
    won_ground_duels = def_ground_duels.loc[def_ground_duels.apply(lambda x:{'id':703} in x.tags, axis = 1)]
    wgd_player = won_ground_duels.groupby(["playerId"]).eventId.count().reset_index()
    wgd_player.rename(columns = {'eventId':'def_ground_duels_won'}, inplace=True)

    air_duels = df.loc[df["subEventName"].isin(["Air duel"])]
    won_air_duels = air_duels.loc[air_duels.apply(lambda x:{'id':703} in x.tags, axis = 1)]
    wad_player = won_air_duels.groupby(["playerId"]).eventId.count().reset_index()
    wad_player.rename(columns = {'eventId':'air_duels_won'}, inplace=True)
    
    duels_won = wgd_player.merge(wld_player, how = "outer", on = ["playerId"]).merge(wad_player, how = "outer", on = ["playerId"])

    return duels_won

duels = defensiveDuelsWon(train)
##################################################################################################################

# CALCULATIONS FOR TURNOVERS #####################################################################################
def calc(values, percentiles):
    lost_pass = values[-1]
    values.pop()
    lost_dribble = values[-1]
    values.pop()
    turnovers = lost_pass + lost_dribble # how many turnovers per 90 min

    lost_pass_perc = percentiles[-1]
    percentiles.pop()
    lost_dribble_perc = percentiles[-1]
    percentiles.pop()

    values_new = values.append(turnovers)
    percentiles_new = percentiles.append(135-turnovers*10)

    return percentiles_new, values_new
##################################################################################################################

# MINUTES PER GAME ###############################################################################################
path = os.path.join(str(pathlib.Path().resolve().parents[0]), 'DM-Football', 'src', 'wyscout', "minutes_played", 'minutes_played_per_game_England.json') 
with open(path) as f:
    minutes_per_game = json.load(f)
minutes_per_game = pd.DataFrame(minutes_per_game)
minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
##################################################################################################################

# SUMMARY ######################################################################################################## 
players = train["playerId"].unique()
summary = pd.DataFrame(players, columns = ["playerId"])
summary = summary.merge(duels, how = "left", on = ["playerId"]).merge(intercept, how = "left", on = ["playerId"]).merge(clear, how = "left", on = ["playerId"]).merge(turnover, how = "left", on = ["playerId"])

summary = minutes.merge(summary, how = "left", on = ["playerId"])
summary = summary.fillna(0)
summary = summary.loc[summary["minutesPlayed"] > 400]
##################################################################################################################

# POSITIONS ######################################################################################################
from tabnanny import verbose

path = os.path.join(str(pathlib.Path().resolve().parents[0]),'DM-Football', 'src', 'wyscout', 'players.json')
with open(path) as f:
    players = json.load(f)
player_df = pd.DataFrame(players)
forwards = player_df.loc[player_df.apply(lambda x: x.role["name"] == "Midfielder", axis = 1)]
forwards.rename(columns = {'wyId':'playerId'}, inplace=True)
to_merge = forwards[['playerId', 'shortName']]
summary = summary.merge(to_merge, how = "inner", on = ["playerId"])

##################################################################################################################

# PER 90 #########################################################################################################
summary_per_90 = pd.DataFrame()
summary_per_90["shortName"] = summary["shortName"]
for column in summary.columns[2:-1]:
    summary_per_90[column + "_per90"] = summary.apply(lambda x: x[column]*90/x["minutesPlayed"], axis = 1)
##################################################################################################################

# S DEFOUR ##########################################################################################
#only his statistics
steven = summary_per_90.loc[summary_per_90["shortName"] == "S. Defour"]
#columns similar together
steven = steven[["loose_duels_won_per90", "def_ground_duels_won_per90", "air_duels_won_per90", "interceptions_per90", "clearances_per90", "lost_dribble_forward_per90", "lost_pass_per90"]]
#take only necessary columns - exclude playerId
per_90_columns_steven = steven.columns[:]
#values to mark on the plot
values_steven = [round(steven[column].iloc[0],2) for column in per_90_columns_steven]
#percentiles
percentiles_steven = [int(stats.percentileofscore(summary_per_90[column], steven[column].iloc[0])) for column in per_90_columns_steven]
# calculate turnovers
calc(values_steven, percentiles_steven)
# rounding
values_steven = [ round(elem, 2) for elem in values_steven ]
percentiles_steven = [ round(elem, 2) for elem in percentiles_steven ]
##################################################################################################################

# A WESTWOOD ##########################################################################################
#only his statistics
ashley = summary_per_90.loc[summary_per_90["shortName"] == "A. Westwood"]
#columns similar together
ashley = ashley[["loose_duels_won_per90", "def_ground_duels_won_per90", "air_duels_won_per90", "interceptions_per90", "clearances_per90", "lost_dribble_forward_per90", "lost_pass_per90"]]
#take only necessary columns - exclude playerId
per_90_columns_ashley = ashley.columns[:]
#values to mark on the plot
values_ashley = [round(ashley[column].iloc[0],2) for column in per_90_columns_ashley]
#percentiles
percentiles_ashley = [int(stats.percentileofscore(summary_per_90[column], ashley[column].iloc[0])) for column in per_90_columns_ashley]
# calculate turnovers
calc(values_ashley, percentiles_ashley)
# rounding
values_ashley = [ round(elem, 2) for elem in values_ashley ]
percentiles_ashley = [ round(elem, 2) for elem in percentiles_ashley ]
##################################################################################################################

# J CORK ##########################################################################################
#only his statistics
jack = summary_per_90.loc[summary_per_90["shortName"] == "J. Cork"]
#columns similar together
jack = jack[["loose_duels_won_per90", "def_ground_duels_won_per90", "air_duels_won_per90", "interceptions_per90", "clearances_per90", "lost_dribble_forward_per90", "lost_pass_per90"]]
#take only necessary columns - exclude playerId
per_90_columns_jack = jack.columns[:]
#values to mark on the plot
values_jack = [round(jack[column].iloc[0],2) for column in per_90_columns_jack]
#percentiles
percentiles_jack = [int(stats.percentileofscore(summary_per_90[column], jack[column].iloc[0])) for column in per_90_columns_jack]
# calculate turnovers
calc(values_jack, percentiles_jack)
# rounding
values_jack = [ round(elem, 2) for elem in values_jack ]
percentiles_jack = [ round(elem, 2) for elem in percentiles_jack ]
##################################################################################################################

# PLOT DEFOUR ####################################################################################################
#list of names on plots
names = ["Loose Ball Duels Won", "Defensive Duels Won", "Air Duels Won", "Interceptions", "Clearances", "Turnovers"]
num = len(names)
slice_colors = ["mediumvioletred"] + ["lightskyblue"] + ["mediumvioletred"] + ["lightskyblue"] + ["mediumvioletred"] + ["lightskyblue"]
text_colors = ["white"] * num
font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
                           "Roboto%5Bwdth,wght%5D.ttf?raw=true"))
font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
                         "RobotoSlab%5Bwght%5D.ttf?raw=true"))
#PIZZA PLOT
baker = PyPizza(
    background_color="white",
    params=names,   
    min_range = None,
    max_range = None,               # list of parameters
    straight_line_color="#000000",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-."            # linestyle for other circles
)
#making pizza for our data
fig, ax = baker.make_pizza(
    percentiles_steven,              # list of values,
    #compare_values=percentiles_steven,
    figsize=(10, 10),      # adjust figsize according to your need
    param_location=110,
    slice_colors = slice_colors,
    value_colors = text_colors,
    value_bck_colors = slice_colors, # where the parameters will be added
    kwargs_slices=dict(
        facecolor="lightskyblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting comparison slices
    kwargs_params=dict(
        color="#000000", fontsize=20,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=15,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="000000",
            boxstyle="round,pad=0.2", lw=1
        )
    )                # values to be used when adding parameter-values  
)

#putting text
texts = baker.get_value_texts()
for i, text in enumerate(texts):
    text.set_text(str(values_steven[i]))

URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/379.png"
img = Image.open(urlopen(URL))
ax_image = add_image(
    img, fig, left=0.4715, bottom=0.4530, width=0.08, height=0.077
)   # these values might differ when you are plotting

# PLOT WESTWOOD ####################################################################################################
#list of names on plots
names = ["Loose Ball Duels Won", "Defensive Duels Won", "Air Duels Won", "Interceptions", "Clearances", "Turnovers"]
num = len(names)
slice_colors = ["mediumvioletred"] + ["lightskyblue"] + ["mediumvioletred"] + ["lightskyblue"] + ["mediumvioletred"] + ["lightskyblue"]
text_colors = ["white"] * num
font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
                           "Roboto%5Bwdth,wght%5D.ttf?raw=true"))
font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
                         "RobotoSlab%5Bwght%5D.ttf?raw=true"))
#PIZZA PLOT
baker = PyPizza(
    background_color="white",
    params=names,   
    min_range = None,
    max_range = None,               # list of parameters
    straight_line_color="#000000",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-."            # linestyle for other circles
)
#making pizza for our data
fig2, ax = baker.make_pizza(
    percentiles_ashley,              # list of values,
    #compare_values=percentiles_steven,
    figsize=(10, 10),      # adjust figsize according to your need
    param_location=110,
    slice_colors = slice_colors,
    value_colors = text_colors,
    value_bck_colors = slice_colors, # where the parameters will be added
    kwargs_slices=dict(
        facecolor="lightskyblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting comparison slices
    kwargs_params=dict(
        color="#000000", fontsize=20,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=15,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="000000",
            boxstyle="round,pad=0.2", lw=1
        )
    )                # values to be used when adding parameter-values  
)

#putting text
texts = baker.get_value_texts()
for i, text in enumerate(texts):
    text.set_text(str(values_ashley[i]))

URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/379.png"
img = Image.open(urlopen(URL))
ax_image = add_image(
    img, fig2, left=0.4715, bottom=0.4530, width=0.08, height=0.077
)   # these values might differ when you are plotting

# PLOT CORK #######################################################################################################
#list of names on plots
names = ["Loose Ball Duels Won", "Defensive Duels Won", "Air Duels Won", "Interceptions", "Clearances", "Turnovers"]
num = len(names)
slice_colors = ["mediumvioletred"] + ["lightskyblue"] + ["mediumvioletred"] + ["lightskyblue"] + ["mediumvioletred"] + ["lightskyblue"]
text_colors = ["white"] * num
font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
                           "Roboto%5Bwdth,wght%5D.ttf?raw=true"))
font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
                         "RobotoSlab%5Bwght%5D.ttf?raw=true"))
#PIZZA PLOT
baker = PyPizza(
    background_color="white",
    params=names,   
    min_range = None,
    max_range = None,               # list of parameters
    straight_line_color="#000000",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-."            # linestyle for other circles
)
#making pizza for our data
fig3, ax = baker.make_pizza(
    percentiles_jack,              # list of values,
    #compare_values=percentiles_steven,
    figsize=(10, 10),      # adjust figsize according to your need
    param_location=110,
    slice_colors = slice_colors,
    value_colors = text_colors,
    value_bck_colors = slice_colors, # where the parameters will be added
    kwargs_slices=dict(
        facecolor="lightskyblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting comparison slices
    kwargs_params=dict(
        color="#000000", fontsize=20,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=15,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="000000",
            boxstyle="round,pad=0.2", lw=1
        )
    )                # values to be used when adding parameter-values  
)

#putting text
texts = baker.get_value_texts()
for i, text in enumerate(texts):
    text.set_text(str(values_jack[i]))

URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/379.png"
img = Image.open(urlopen(URL))
ax_image = add_image(
    img, fig3, left=0.4715, bottom=0.4530, width=0.08, height=0.077
)   # these values might differ when you are plotting

##################################################################################################################

# ASSIGNMENT 2 #

# was a secondary file but merged them togehter for getting all plots at the same time, therefore the code is repeatetive

# IMPORTS ########################################################################################################
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

country = "France"

# GET DATA #######################################################################################################
path = os.path.join(str(pathlib.Path().resolve()), 'src', 'wyscout', 'events', 'events_' + f'{country}' + '.json') 
with open(path) as f: 
    data = json.load(f) 
train = pd.DataFrame(data) 

train = train.loc[train.apply (lambda x: len(x.positions) == 2, axis = 1)]
##################################################################################################################

# TURNOEVERS #####################################################################################################
def turnovers(df):
    dribble_forward = df.loc[df["subEventName"] == "Ground attacking duel"]
    lost_dribble_forward = dribble_forward.loc[dribble_forward.apply(lambda x:{'id':701} in x.tags, axis = 1)]
    ldf = lost_dribble_forward.groupby(["playerId"]).eventId.count().reset_index()
    ldf.rename(columns = {'eventId':'lost_dribble_forward'}, inplace=True)
    
    passes = df.loc[df["eventName"] == "Pass"]
    lost_pass = passes.loc[passes.apply(lambda x:{'id':1802} in x.tags, axis = 1)]
    lp = lost_pass.groupby(["playerId"]).eventId.count().reset_index()
    lp.rename(columns = {'eventId':'lost_pass'}, inplace=True)

    turnovers_made = ldf.merge(lp, how = "outer", on = ["playerId"])

    return turnovers_made

turnover = turnovers(train)
##################################################################################################################

# CLEARANCES #####################################################################################################
def clearance(df):
    clearance = df.loc[df["subEventName"] == "Clearance"]
    clearing_player = clearance.groupby(["playerId"]).eventId.count().reset_index()
    clearing_player.rename(columns = {'eventId':'clearances'}, inplace=True)

    return clearing_player

clear = clearance(train)
##################################################################################################################

# INTERCEPTIONS ##################################################################################################
def interception(df):
    opp_pass = df.loc[df["eventName"] == "Pass"]
    interception = opp_pass.loc[opp_pass.apply(lambda x:{'id':1401} in x.tags, axis = 1)]
    intercepting_player = interception.groupby(["playerId"]).eventId.count().reset_index()
    intercepting_player.rename(columns = {'eventId':'interceptions'}, inplace=True)

    return intercepting_player

intercept = interception(train)
##################################################################################################################

# DUELS WON ###################################################################################################### 
def defensiveDuelsWon(df):
    loose_duels = df.loc[df["subEventName"] == "Ground loose ball duel"]
    won_loose_duels = loose_duels.loc[loose_duels.apply(lambda x:{'id':703} in x.tags, axis = 1)]
    wld_player = won_loose_duels.groupby(["playerId"]).eventId.count().reset_index()
    wld_player.rename(columns = {'eventId':'loose_duels_won'}, inplace=True)
    
    def_ground_duels = df.loc[df["subEventName"].isin(["Ground defending duel"])]
    won_ground_duels = def_ground_duels.loc[def_ground_duels.apply(lambda x:{'id':703} in x.tags, axis = 1)]
    wgd_player = won_ground_duels.groupby(["playerId"]).eventId.count().reset_index()
    wgd_player.rename(columns = {'eventId':'def_ground_duels_won'}, inplace=True)

    air_duels = df.loc[df["subEventName"].isin(["Air duel"])]
    won_air_duels = air_duels.loc[air_duels.apply(lambda x:{'id':703} in x.tags, axis = 1)]
    wad_player = won_air_duels.groupby(["playerId"]).eventId.count().reset_index()
    wad_player.rename(columns = {'eventId':'air_duels_won'}, inplace=True)
    
    duels_won = wgd_player.merge(wld_player, how = "outer", on = ["playerId"]).merge(wad_player, how = "outer", on = ["playerId"])

    return duels_won

duels = defensiveDuelsWon(train)

##################################################################################################################

# CALCULATIONS FOR TURNOVERS #####################################################################################
def calc(values, percentiles):
    lost_pass = values[-1]
    values.pop()
    lost_dribble = values[-1]
    values.pop()
    turnovers = lost_pass + lost_dribble # how many turnovers per 90 min

    percentiles.pop()
    percentiles.pop()

    values_new = values.append(turnovers)
    percentiles_new = percentiles.append(turnovers*10) # multiplying by 10 to get the right scale

    return percentiles_new, values_new
##################################################################################################################

# MINUTES PER GAME ###############################################################################################
path = os.path.join(str(pathlib.Path().resolve().parents[0]),'DM-Football', 'src', 'wyscout', "minutes_played", 'minutes_played_per_game_' + f'{country}' + '.json') 
with open(path) as f:
    minutes_per_game = json.load(f)
minutes_per_game = pd.DataFrame(minutes_per_game)
minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
##################################################################################################################

# SUMMARY ######################################################################################################## 
players = train["playerId"].unique()
summary = pd.DataFrame(players, columns = ["playerId"])
summary = summary.merge(duels, how = "left", on = ["playerId"]).merge(intercept, how = "left", on = ["playerId"]).merge(clear, how = "left", on = ["playerId"]).merge(turnover, how = "left", on = ["playerId"])

summary = minutes.merge(summary, how = "left", on = ["playerId"])
summary = summary.fillna(0)
summary = summary.loc[summary["minutesPlayed"] > 400]
##################################################################################################################

# POSITIONS ######################################################################################################
from tabnanny import verbose

path = os.path.join(str(pathlib.Path().resolve().parents[0]),'DM-Football', 'src', 'wyscout', 'players.json')
with open(path) as f:
    players = json.load(f)
player_df = pd.DataFrame(players)
forwards = player_df.loc[player_df.apply(lambda x: x.role["name"] == "Midfielder", axis = 1)]
forwards.rename(columns = {'wyId':'playerId'}, inplace=True)
to_merge = forwards[['playerId', 'shortName']]
summary = summary.merge(to_merge, how = "inner", on = ["playerId"])

#print(summary.head(50))

##################################################################################################################

# PER 90 #########################################################################################################
summary_per_90 = pd.DataFrame()
summary_per_90["shortName"] = summary["shortName"]
for column in summary.columns[2:-1]:
    summary_per_90[column + "_per90"] = summary.apply(lambda x: x[column]*90/x["minutesPlayed"], axis = 1)
##################################################################################################################

# Player investigated ##########################################################################################
#only his statistics
player = summary_per_90.loc[summary_per_90["shortName"] == "A. Zambo Anguissa"]
#columns similar together
player = player[["loose_duels_won_per90", "def_ground_duels_won_per90", "air_duels_won_per90", "interceptions_per90", "clearances_per90", "lost_dribble_forward_per90", "lost_pass_per90"]]
#take only necessary columns - exclude playerId
per_90_columns_player = player.columns[:]
#values to mark on the plot
values_player = [round(player[column].iloc[0],2) for column in per_90_columns_player]
#percentiles
percentiles_player = [int(stats.percentileofscore(summary_per_90[column], player[column].iloc[0])) for column in per_90_columns_player]
# calculate turnovers
calc(values_player, percentiles_player)
# rounding
values_player = [ round(elem, 2) for elem in values_player ]
percentiles_player = [ round(elem, 2) for elem in percentiles_player ]
##################################################################################################################

# PLOT ###########################################################################################################
#list of names on plots
names = ["Loose Ball Duels Won", "Defensive Duels Won", "Air Duels Won", "Interceptions", "Clearances", "Turnovers"]
num = len(names)
slice_colors = ["deepskyblue"] + ["khaki"] + ["deepskyblue"] + ["khaki"]  + ["deepskyblue"] + ["khaki"] 
text_colors = ["white"] * num
font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
                           "Roboto%5Bwdth,wght%5D.ttf?raw=true"))
font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
                         "RobotoSlab%5Bwght%5D.ttf?raw=true"))
#PIZZA PLOT
baker = PyPizza(
    background_color="white",
    params=names,   
    min_range = None,
    max_range = None,               # list of parameters
    straight_line_color="#000000",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-."            # linestyle for other circles
)
#making pizza for our data
fig4, ax = baker.make_pizza(
    percentiles_player,              # list of values,
    #compare_values=percentiles_steven,
    figsize=(10, 10),      # adjust figsize according to your need
    param_location=110,
    slice_colors = slice_colors,
    value_colors = text_colors,
    value_bck_colors = slice_colors, # where the parameters will be added
    kwargs_slices=dict(
        facecolor="lightskyblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting comparison slices
    kwargs_params=dict(
        color="#000000", fontsize=20,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=15,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="000000",
            boxstyle="round,pad=0.2", lw=1
        )
    )                # values to be used when adding parameter-values  
)

#putting text
texts = baker.get_value_texts()
for i, text in enumerate(texts):
    text.set_text(str(values_player[i]))

# add image
from urllib.request import urlopen
from PIL import Image
import matplotlib.image as mpimg

URL = "https://i.imgur.com/BSbHxME.png"
img = Image.open(urlopen(URL))
#img = mpimg.imread('anguissa_crop.png')
ax_image = add_image(
    #img, fig, left=0.385, bottom=0.38, width=0.25, height=0.25
    img, fig4, left=0.4715, bottom=0.4530, width=0.08, height=0.077
)   # these values might differ when you are plotting

plt.show()

##################################################################################################################

# ASSIGNMENT 2 #

# IMPORTS ########################################################################################################
import pandas as pd
import json
#opening data
import os
import pathlib
import warnings 

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

country2 = "France"

# GET DATA #######################################################################################################
path = os.path.join(str(pathlib.Path().resolve()), 'src', 'wyscout', 'events', 'events_' + f'{country2}' + '.json') 
with open(path) as f:
    data = json.load(f) 
train = pd.DataFrame(data) 

train = train.loc[train.apply (lambda x: len(x.positions) == 2, axis = 1)]
##################################################################################################################

# INTERCEPTIONS ##################################################################################################
def interception(df):
    opp_pass = df.loc[df["eventName"] == "Pass"]
    interception = opp_pass.loc[opp_pass.apply(lambda x:{'id':1401} in x.tags, axis = 1)]
    intercepting_player = interception.groupby(["playerId"]).eventId.count().reset_index()
    intercepting_player.rename(columns = {'eventId':'interceptions'}, inplace=True)

    return intercepting_player

intercept = interception(train)
##################################################################################################################

# GROUP BY PLAYER ################################################################################################
interceptions_by_player = intercept.groupby(["playerId"])["interceptions"].sum().reset_index()
##################################################################################################################

# MERGING PLAYER NAME ############################################################################################
path = os.path.join(str(pathlib.Path().resolve().parents[0]),'DM-Football', 'src', 'wyscout', 'players.json')
with open(path) as f:
    players = json.load(f)
player_df = pd.DataFrame(players)
player_df = player_df.loc[player_df.apply(lambda x: x.role["name"] == "Midfielder", axis = 1)]
player_df.rename(columns = {'wyId':'playerId'}, inplace=True)
player_df["role"] = player_df.apply(lambda x: x.role["name"], axis = 1)
to_merge = player_df[['playerId', 'firstName', 'lastName', 'role']]

summary = interceptions_by_player.merge(to_merge, how = "inner", on = ["playerId"])
##################################################################################################################

# MINUTES PER GAME ###############################################################################################
path = os.path.join(str(pathlib.Path().resolve().parents[0]),'DM-Football', 'src', 'wyscout', "minutes_played", 'minutes_played_per_game_' + f'{country}' + '.json') 
with open(path) as f:
    minutes_per_game = json.load(f)
minutes_per_game = pd.DataFrame(minutes_per_game)
minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
##################################################################################################################

# SUMMARY ######################################################################################################## 
summary = minutes.merge(summary, how = "left", on = ["playerId"])
summary = summary.fillna(0)
summary = summary.loc[summary["minutesPlayed"] > 400]
##################################################################################################################

# PER 90 #########################################################################################################
summary["interceptions_per_90"] = summary["interceptions"]*90/summary["minutesPlayed"]
##################################################################################################################

# FINAL ##########################################################################################################
summary = summary[['firstName', 'lastName', 'interceptions_per_90']].sort_values(by='interceptions_per_90', ascending=False)
##################################################################################################################
print(summary.head(10))