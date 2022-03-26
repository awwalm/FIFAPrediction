# TrimFeatures.py

import sqlite3
from sqlite3 import Error
import pandas as pd
import os

db_file = os.getcwd() + "/Datasets/fifa-20-complete-player-dataset/Goalkeepers_Outfield_Players.sqlite"

conn = sqlite3.connect(db_file, isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)

# creating the Goalkeepers string queries

gkquery1 = "sofifa_id, short_name, age, overall, value_eur, wage_eur, release_clause_eur, team_position,"
gkquery2 = "weight_kg, movement_reactions, goalkeeping_handling, goalkeeping_kicking, gk_handling,"
gkquery3 = "goalkeeping_positioning, goalkeeping_reflexes, goalkeeping_diving, gk_diving, gk_reflexes"
gkquery = gkquery1 + gkquery2 + gkquery3

# creaing the Outfield Players string queries

outquery1 = "sofifa_id, short_name, age, overall, value_eur, wage_eur, release_clause_eur,"
outquery2 = "team_position, skill_ball_control, dribbling, attacking_finishing, passing,"
outquery3 = "shooting, movement_reactions, mentality_vision, skill_dribbling, power_strength, pace"
outquery = outquery1 + outquery2 + outquery3

# Reading from sqlite database and converting each table to CSV
#  and limiting the rows to 500 and 1000 respectively

gk_db_df = pd.read_sql_query("SELECT " + gkquery + " FROM Goalkeepers LIMIT 500", conn)
gk_db_df.to_csv('Datasets/Goalkeepers_features.csv', index=False)

outfield_db_df = pd.read_sql_query("SELECT " + outquery + " FROM Outfield_Players LIMIT 1000", conn)
outfield_db_df.to_csv('Datasets/Outfield_Players_features.csv', index=False)

