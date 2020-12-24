#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 23 01:37:48 2020
@author: plkn
"""

# Imports
from datetime import datetime, timedelta
import requests
import locale
import time
import pandas as pd
import math
from bs4 import BeautifulSoup
import joblib
import os

# current directory with data file
data_path = os.path.abspath(f"{os.getcwd()}/data/{os.path.basename(__file__)}.joblib")

# DataFrame columns
columns = [
    "datetime",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "minutes_since_thread_start",
    "page_number",
    "post_number_all",
    "post_number_tom",
    "message_number_total",
    "message_number_post",
    "message_length",
    "message",
]

# Init DataFrame
df = pd.DataFrame([], columns=columns)

# Set sleep time between scrapes
sleep_time = 1

# switch to german locale
locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")

# forum page index
# full range: 1, 361
pages = range(1, 361)

# Some counters
post_number_all = 0
post_number_tom = 0
message_number_total = 0

# Iterate pages
for page in pages:
    print(f"{datetime.now().strftime('%H:%M:%S')} -> Getting Thread Page -> {page}")

    # thread url
    url = f"https://www.civforum.de/showthread.php?103910-Lustige-Story-von-tom-bombadil/page{page}"

    # Access site with requests
    response = requests.get(url)

    # Parse html to soup
    soup = BeautifulSoup(response.content, features="html.parser")

    # Get the posts
    posts = soup.find_all("li", {"class": "postcontainer"})

    # Iterate posts of page (postcontainer)
    for post in posts:

        # absolute count
        post_number_all += 1

        # Check if post is a guest post
        if len(post.find_all("span", {"class": "username guest"})) > 0:
            continue

        # check if the post belongs to tombombadil
        username = (
            post.find_all("a", {"class": "username"})[0].find_all("strong")[0].text
        )
        if username != "tom.bombadil":
            continue

        # absolute count for tom
        post_number_tom += 1

        # get the date and time, parse into date
        post_date_container = post.find_all("span", {"class": "postdate"})[0].find_all(
            "span", {"class": "date"}
        )[0]

        # Get date string
        post_date = post_date_container.text.split(",")[0]

        # Handle special cases
        if post_date == "Heute":
            post_date = datetime.now().strftime("%d. %B %Y")
        elif post_date == "Gestern":
            post_date = (datetime.now() - timedelta(days=1)).strftime("%d. %B %Y")

        # get time, always there
        post_time = post_date_container.find_all("span", {"class": "time"})[0].text

        # Get a datetime
        date = datetime.strptime(f"{post_date} {post_time}", "%d. %B %Y %H:%M")

        # Get start of thread time
        if post_number_tom == 1:
            date_start = date

        # remove all citations from the post
        for quote in post.find_all("div", {"class": "quote_container"}):
            quote.decompose()

        # parse the post content into an ordered message list
        post_content = post.find_all("div", {"class": "content"})[0].find_all(text=True)
        messages = list(
            filter(lambda m: m != "", list(map(lambda m: m.strip(), post_content)))
        )

        # Iterate messages in post
        for message_number, message in enumerate(messages, start=1):

            # Exclude messages with length <= 3
            # TODO

            message_number_total += 1
            data = {
                "datetime": date,
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "hour": date.hour,
                "minute": date.minute,
                "minutes_since_thread_start": math.floor(
                    (date - date_start).total_seconds() / 60
                ),
                "page_number": page,
                "post_number_all": post_number_all,
                "post_number_tom": post_number_tom,
                "message_number_total": message_number_total,
                "message_number_post": message_number,
                "message_length": len(message),
                "message": message,
            }

            # Append dictionary to df
            df_tmp = pd.DataFrame([data], columns=data.keys())
            df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

    time.sleep(sleep_time)

# Save data
joblib.dump(df, data_path)
