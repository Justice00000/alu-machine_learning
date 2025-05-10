#!/usr/bin/env python3
<<<<<<< HEAD


""" Return list of ships"""

import requests
import sys
import time


if __name__ == "__main__":
    res = requests.get(sys.argv[1])

    if res.status_code == 403:
        rate_limit = int(res.headers.get('X-Ratelimit-Reset'))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))
        # get remaining rate

    elif res.status_code == 404:
        print("Not found")
    elif res.status_code == 200:
        res = res.json()
        print(res['location'])
=======
"""
Uses the GitHub API to print the location of a specific user,
where user is passed as first argument of the script with full API URL

ex) "./2-user_location.py https://api.github.com/users/holbertonschool"
"""


import requests
from sys import argv
from time import time


if __name__ == "__main__":
    if len(argv) < 2:
        raise TypeError(
            "Input must have the full API URL passed in as an argument: {}{}".
            format('ex. "./2-user_location.py',
                   'https://api.github.com/users/holbertonschool"'))
    try:
        url = argv[1]
        results = requests.get(url)
        if results.status_code == 403:
            reset = results.headers.get('X-Ratelimit-Reset')
            waitTime = int(reset) - time()
            minutes = round(waitTime / 60)
            print('Reset in {} min'.format(minutes))
        else:
            results = results.json()
            location = results.get('location')
            if location:
                print(location)
            else:
                print('Not found')
    except Exception as err:
        print('Not found')
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
