#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 01:37:48 2020

@author: plkn
"""

# Imports
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

# Set url
url = 'https://www.civforum.de/showthread.php?103910-Lustige-Story-von-tom-bombadil/page1'

# Access site with requests
response = requests.get(url)

# Parse html to soup
soup = BeautifulSoup(response.text, 'html.parser')

