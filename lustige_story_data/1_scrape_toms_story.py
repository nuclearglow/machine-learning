#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 23 01:37:48 2020
@author: nuky, plkn
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

# current directory and data file name
data_path = os.path.abspath(f"{os.getcwd()}/data/tom_scrapedata.joblib")

# DataFrame columns
columns = [
    "datetime",
    "year",
    "month",
    "day",
    "weekday",
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

# Set sleep time between scrapes
sleep_time = 1

# switch to german locale
locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")

# * Scrape the remaining range and save

# get the last page from the link on the first page
url = f"https://www.civforum.de/showthread.php?103910-Lustige-Story-von-tom-bombadil/page1"
response = requests.get(url)
soup = BeautifulSoup(response.content, features="html.parser")
last_page_container = soup.find_all("span", {"class": "first_last"})[0]
last_page_link = last_page_container.find_all("a")[0].attrs["href"]
last_page = int(last_page_link .split("page")[1].split("&")[0])

# Init DataFrame or load existing
if os.path.exists(data_path):
    df = joblib.load(data_path)
    # Load counters from df
    df_last_page = df["page_number"].iloc[-1]
    last_date = df["datetime"].iloc[-1]
    date_start = df["datetime"][0]
    pages = range(df_last_page, last_page + 1)

    # Some counters - increment from last entries in dataframe
    post_number_all = df["post_number_all"].iloc[-1]
    post_number_tom = df["post_number_tom"].iloc[-1]
    message_number_total = df["message_number_total"].iloc[-1]

else:
    df = pd.DataFrame([], columns=columns)
    df_last_page = 1
    last_date = None
    # Some counters - initialize
    post_number_all = 0
    post_number_tom = 0
    message_number_total = 0

# forum page index
pages = range(df_last_page, last_page + 1)

print(f"Newsflash: Scraping New Pages {df_last_page} to {last_page}")

# Iterate pages
for page in pages:
    print(f"{datetime.now().strftime('%H:%M:%S')} -> Getting Thread Page -> {page}")

    # page url
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

        if last_date != None and date <= last_date:
            print("Post already in database. Skipping.")
            continue

        # absolute count for tom
        post_number_tom += 1

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
            if len(message) > 3:

                message_number_total += 1
                data = {
                    "datetime": date,
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "weekday": date.weekday(),
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
