#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: DDTIdownload.py
@datatime: 8/15/2023 11:52 AM
"""

import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


# DOWNLOAD DATA
file_id = r'1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R'
# DESTINATION FILE ON YOUR DISK
destination = r'/DDTI.zip'
download_file_from_google_drive(file_id, destination)
# DOWNLOAD WEIGHT
file_id = r'16CldjuNztEZp2E3SJzXTHp--JggiNCMX'
# DESTINATION FILE ON YOUR DISK
destination = r'/DDTI_weight.zip'
download_file_from_google_drive(file_id, destination)
