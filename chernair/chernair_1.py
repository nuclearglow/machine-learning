#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import os

# Path vars
path_data = '/home/plkn/repos/machine-learning/chernair/'




data = pd.read_csv(os.path.join(path_data, 'CHERNAIR.csv'))

