#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import os
import geopy.distance
import datetime
from pytz import timezone

# Path vars
path_data = '/home/plkn/repos/machine-learning/chernair/'

# Read data
data = pd.read_csv(os.path.join(path_data, 'CHERNAIR.csv'))
data.rename(columns={'Y':'lat', 'X':'lng'}, inplace=True)


# Chernobyl explosion coordinates
chern_lat_lng = (51.386998452, 30.092666296)

# Create chernobyl explosion time in UTC
cherntime_naive = datetime.datetime(1986, 4, 26, 1, 23, 40)
chern_tz = timezone('Europe/Kiev')
cherntime_aware = chern_tz.localize(cherntime_naive)

# Get distance from explosion in km
def get_distance_from_chernobyl_explosion(row):
    return geopy.distance.distance(tuple(row), chern_lat_lng).km
data['cherndist'] = data[['lat', 'lng']].apply(get_distance_from_chernobyl_explosion, axis=1)

# Get time from explosion in seconds
def get_seconds_from_chernobyl_explosion(row):
    if row[1].count(':') == 1:
        row[1] = f'{row[1]}:00'
    year, month, day = [int(x) for x in row[0].split('/')]
    year += 1900
    hour, minute, second = [int(x) for x in row[1].split(':')]
    incday = False
    if hour == 24:
        hour = 0
        incday = True
    sampletime_naive = datetime.datetime(year, month, day, hour, minute, second)
    if incday:
        sampletime_naive = sampletime_naive + datetime.timedelta(days=1)
    sample_tz = timezone('CET')
    sampletime_aware = sample_tz.localize(sampletime_naive)
    return (sampletime_aware - cherntime_aware).total_seconds()
data['cherntime'] = data[['Date', 'End of sampling']].apply(get_seconds_from_chernobyl_explosion, axis=1)



