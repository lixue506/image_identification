#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import requests


def get_token():
    get_token_url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": "4abf0b6b06284d109866a42cbaae166f",
        "client_secret": "c1d2e3dfdba14fed9390ea774ca764d9",
    }
    res = requests.get(get_token_url, params=params).json()
    print(res)
    return res["access_token"]


if __name__ == '__main__':
    access_token = get_token()
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"
    with open("bus.jpg", mode='rb')as f:
        image = base64.b64encode(f.read())
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        "access_token": access_token,
        "image": image
    }
    res = requests.post(url, headers=headers, data=data).json()["words_result"]
    print(res["number"])
