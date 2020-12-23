#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 23 01:37:48 2020
@author: plkn, nuky
"""
import os
import joblib
import locale
import requests
import tinydb
import time
import math
from bs4 import BeautifulSoup

# A tiny db
db_path = os.path.abspath(f"{os.getcwd()}/data/words.json")
db = tinydb.TinyDB(db_path)

# Filename of lustige story
filename = "data_extraction.py.joblib"

# Set sleep time between scrapes
sleep_time = 1

# switch to german locale
locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")

# current directory with data file
data_path = os.path.abspath(f"{os.getcwd()}/data/{filename}")

# Load the lustige story
df = joblib.load(data_path)

# Function for requests
def get_word_metadata(word):
    # word is handled by title and lowercased
    word_lower, word_title = word.lower(), word.title()
    # db lookup, deliver immediately
    q = tinydb.Query()
    results = db.search(q.word == word_lower)
    if results and len(results) > 0:
        return results[0]

    print(f"Querying UniLeipzig for {word_lower} and {word_title}")

    results = []
    for word in [word_lower, word_title]:
        # dictionary url
        url = f"https://corpora.uni-leipzig.de/en/res?corpusId=deu_newscrawl-public_2018&word={word}"

        # Access site with requests
        time.sleep(sleep_time)
        response = requests.get(url)
        if response.status_code == 404:
            print(f"Word {word} not found :-(")
            word_metadata = {
                "word": word_lower,
                "request": word,
                "freq_rank": math.inf,
                "freq_class": math.inf,
                "word_classes": [],
            }
            results.append(word_metadata)
            continue

        # Parse html to soup
        soup = BeautifulSoup(response.content, features="html.parser")

        # Get the wordbox
        wordbox = soup.find_all("div", {"id": "wordBox"})[0]

        # Get the header stats
        stats_raw = wordbox.find_all("div", {"class": "panel-heading"})[0].find_all(
            text=True
        )
        stats = list(
            filter(lambda m: m != "", list(map(lambda m: m.strip(), stats_raw)))
        )

        # Get rank and frequency class
        rank = int(stats[3].split(":")[1].strip().replace(",", ""))
        frequency_class = int(stats[4].split(":")[1].strip().replace(",", ""))

        panel_body = soup.find_all("div", {"class": "panel-body"})[0]
        part_container = panel_body.find(string="Part of speech:")
        part = (
            part_container.parent.next_sibling.strip()
            if part_container and len(part_container) > 0
            else ""
        )
        word_classes = (
            list(map(lambda w: w.strip(), part.split(","))) if "," in part else [part]
        )

        word_metadata = {
            "word": word_lower,
            "request": word,
            "freq_rank": rank,
            "freq_class": frequency_class,
            "word_classes": word_classes,
        }
        results.append(word_metadata)

    # check results and insert into db and return
    selected_index = 0 if results[0]["freq_rank"] < results[1]["freq_rank"] else 1
    final_metadata = {
        "word": word_lower,
        "request": results[selected_index]["request"],
        "freq_rank": results[selected_index]["freq_rank"],
        "freq_class": results[selected_index]["freq_class"],
        "word_classes": list(
            set(
                list(
                    filter(
                        lambda wc: wc != "",
                        results[0]["word_classes"] + results[1]["word_classes"],
                    )
                )
            )
        ),
    }

    # persist to database
    db.insert(final_metadata)

    return final_metadata


print(get_word_metadata(word="dem"))
