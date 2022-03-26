# PreProcessSQLite.py

import sqlite3
from sqlite3 import Error
import pandas as pd
import os

# Dataset path
db_file = os.getcwd() + "/Datasets/european-soccer/database.sqlite"

conn = sqlite3.connect(db_file, isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)

# Reading from sqlite database and converting each table to CSV

country_db_df = pd.read_sql_query("SELECT * FROM Country", conn)
country_db_df.to_csv('Country.csv', index=False)

league_db_df = pd.read_sql_query("SELECT * FROM League", conn)
league_db_df.to_csv('League.csv', index=False)

match_db_df = pd.read_sql_query("SELECT * FROM Match", conn)
match_db_df.to_csv('Match.csv', index=False)

player_db_df = pd.read_sql_query("SELECT * FROM Player", conn)
player_db_df.to_csv('Player.csv', index=False)

player_attr_db_df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
player_attr_db_df.to_csv('Player_Attributes.csv', index=False)

team_db_df = pd.read_sql_query("SELECT * FROM Team", conn)
team_db_df.to_csv('Team.csv', index=False)

team_attr_db_df = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
team_attr_db_df.to_csv('Team_Attributes.csv', index=False)
