# PlayerGroup.py

import sqlite3
from sqlite3 import Error
import pandas as pd
import os

# DATASET PATH
db_file = os.getcwd() + "/Datasets/fifa-20-complete-player-dataset/players_20_sqlite.sqlite"

conn = sqlite3.connect(db_file, isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)

# Reading from sqlite database and converting each table to 

gk_db_df = pd.read_sql_query(
    "SELECT * FROM players_20_experimental WHERE team_position = 'GK' ", conn)
gk_db_df.to_csv('Datasets/Goalkeepers.csv', index=False)

outfield_db_df = pd.read_sql_query(
    "SELECT * FROM players_20_experimental WHERE team_position != 'GK' ", conn)
outfield_db_df.to_csv('Datasets/Outfield_Players.csv', index=False)

