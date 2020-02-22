#!/usr/bin/env python
import os
import tarfile
from six.moves import urllib

# download: https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.tgz
DOWNLOAD_URL = 'https://github.com/ageron/handson-ml/raw/master/'
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_URL + HOUSING_PATH + "/housing.tgz"


# download housing data, put into housing path,
def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    # create housing path if not exist
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    # build path to file
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # download and save
    urllib.request.urlretrieve(housing_url, tgz_path)
    # extract to housing path
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

