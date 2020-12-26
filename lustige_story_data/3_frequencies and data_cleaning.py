#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:11:32 2020

@author: nuky, plkn
"""
import os


# Load last scrape dataframe
path_data = os.getcwd() + "/data/"
file_list = os.listdir(path_data)
scrape_files = [x for x in file_list if x.startswith("tom_scrapedata_")]
dts = []
for file in scrape_files:
    dt = file.strip(".joblib").split("_")[2:]
    dt = dt[0].split("-") + dt[1].split("-")
    dt = [dt[i] for i in [2, 1, 0, 3, 4, 5]]
    dts.append(datetime.datetime(*map(int, dt)))
file_time_string = max(dts).strftime("%d-%m-%Y_%H-%M-%S")
filename = f"tom_scrapedata_{file_time_string}.joblib"


# Save data
filename_out = f"data_words_scraped_{file_time_string}.joblib"
data_path = os.path.abspath(f"{os.getcwd()}/data/{filename_out}")
joblib.dump(df_words, data_path)



# TODO: Create ordered table with word frequencies, add frequency info to dataset

# TODO: Visual inspection. Missing info for important (frequent nouns) words.

# TODO: Plot some descriptive statistics