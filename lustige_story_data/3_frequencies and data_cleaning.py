#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:11:32 2020

@author: nuky, plkn
"""
# Imports
import os
import re
import itertools
import numpy as np
import joblib
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt

# Load last scrape dataframe
path_words_data = f"{os.getcwd()}/data/tom_wordsdata.joblib"
df_words = joblib.load(path_words_data)

# Add frequencies as column
word_freqs = df_words["word_db"].value_counts().to_dict()
df_words["occurences"] = 0
df_words["occurences"] = df_words["word_db"].map(word_freqs.get)

# Get unique words and corresponding frequencies
df_unique = (
    df_words[["word_db", "word_classes", "occurences"]]
    .drop_duplicates(subset=["word_db"])
    .sort_values(by="occurences", ascending=False)
)

# Get words with no detected classes
df_empty = df_unique[df_unique["word_classes"].apply(lambda x: len(x)) == 0]

# Keep only words with occurences > 5
df_empty = df_empty[df_empty["occurences"] > 5].reset_index()

# Manual Index of words we want to keep from empty
words_to_keep_empty_index = [
    1,
    2,
    3,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    25,
    26,
    27,
    29,
    30,
    32,
    34,
    35,
    36,
    37,
    38,
    40,
    41,
    42,
    45,
    46,
    48,
    49,
    50,
    51,
    52,
    55,
    56,
    61,
    62,
    63,
    64,
    65,
    67,
    68,
    69,
    71,
    73,
    75,
    77,
    78,
    79,
    82,
    83,
    85,
    86,
    87,
    88,
    90,
    91,
    93,
    94,
    95,
    96,
    97,
    99,
    100,
    101,
    102,
    103,
    104,
    107,
    108,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    118,
    120,
    121,
    122,
    124,
    125,
    127,
    128,
    129,
    131,
    134,
    136,
    137,
    138,
    139,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    159,
    160,
    161,
    162,
    164,
    165,
    167,
    168,
    169,
    170,
    173,
    175,
    177,
    182,
    183,
    184,
    188,
    189,
    190,
    192,
    193,
    197,
    198,
    199,
]

# Bool vector for words to keep by visual inspection
idx1 = df_words["word_db"].isin(df_empty["word_db"][words_to_keep_empty_index]).values

# Bool vector for words to keep with correct class
idx2 = (
    df_words["word_classes"]
    .apply(lambda x: set(x).issubset({"Noun", "Verb", "Proper noun"}))
    .values
)

# Bool vector for occurences >= 5
idx3 = (df_words["occurences"] >= 5).values

# Bool vector for non empty strings
idx4 = df_words["word"].apply(lambda x: len(x.strip()) > 0).values

# Bool vector for non number strings
numbers_only_regexp = re.compile("^\d+$")
idx5 = df_words["word"].apply(lambda x: numbers_only_regexp.match(x) is None).values

# Combine conditions
idx = (idx1 | idx2) & idx3 & idx4 & idx5

# Select relevant words
df_chosen_words = df_words.iloc[idx]

# Visualization via wordcloud
w = " ".join(df_chosen_words["word_db"].to_list())
wordcloud = WordCloud(
    width=2048, height=1024, background_color="black", min_font_size=10
).generate(w)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Get chosen words as list
list_of_chosen_words = list(set(list(df_chosen_words["word_db"])))

# Initialize an adjacency Matrix
adjacency_matrix = np.zeros((len(list_of_chosen_words), len(list_of_chosen_words)))

# Get number of posts
n_posts = df_words["post_number_tom"].max()

# dictionary of edge weights
pair_dict = {}

# iterate posts
for post_index in range(1, n_posts + 1):
    # get all chosen words in post
    chosen_words_in_post = df_chosen_words[
        df_chosen_words["post_number_tom"] == post_index
    ]["word_db"].values
    # create neighboring pairs
    pairs = [
        (i, j)
        for i, j in itertools.product(chosen_words_in_post, chosen_words_in_post)
        if i != j
    ]
    # eliminate identities [a,b] <-> [b,a]
    pairs = {tuple(pair) for pair in map(sorted, pairs)}
    # increase weight for each pair
    for pair in pairs:
        if pair in pair_dict:
            pair_dict[pair] += 1
        else:
            pair_dict[pair] = 1

# create a graph
G = nx.Graph()

G.add_edges_from(
    [(pair[0], pair[1], {"weight": pair_dict[pair]}) for pair in pair_dict if pair_dict[pair] > 10]
)
# export graph to graphml
# graph_path = file_words.replace(".joblib", ".graphml")
# nx.write_graphml(G, graph_path)

nx.draw_networkx(G, with_labels=True)


# TODO: Create ordered table with word frequencies, add frequency info to dataset

# TODO: Visual inspection. Missing info for important (frequent nouns) words.

# TODO: Plot some descriptive statistics
