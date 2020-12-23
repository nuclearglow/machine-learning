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
import pandas
import math
from bs4 import BeautifulSoup

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
df = pandas.DataFrame([], columns=columns)

# Set sleep time between scrapes
sleep_time = 3

# switch to german locale
locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")

# forum page index
pages = range(1, 5)
username = "tom.bombadil"

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
    post_number_tom = 0
    message_number_total = 0
    for post_number_all, post in enumerate(posts, start=1):

        # Check if post is a guest post
        if len(post.find_all("span", {"class": "username guest"})) > 0:
            continue

        # check if the post belongs to tombombadil
        username = (
            post.find_all("a", {"class": "username"})[0].find_all("strong")[0].text
        )
        if username != "tom.bombadil":
            continue

        # relative count
        post_number_tom += 1

        # get the date and time, parse into date
        # TODO: 'heute' und 'gestern'
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
        if post_number_all == 1:
            date_start = date

        # parse the post content into an ordered message list
        post_content = post.find_all("div", {"class": "content"})[0].find_all(text=True)
        messages = list(
            filter(lambda m: m != "", list(map(lambda m: m.strip(), post_content)))
        )

        # Iterate messages in post
        for message_number, message in enumerate(messages, start=1):
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
            df.append(data, ignore_index=True)

    time.sleep(sleep_time)
