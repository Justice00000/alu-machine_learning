#!/usr/bin/env python3
<<<<<<< HEAD
"""Pipeline Api"""
import requests


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    rocket_dict = {"5e9d0d95eda69955f709d1eb": 0}

    for launch in r.json():
        if launch["rocket"] in rocket_dict:
            rocket_dict[launch["rocket"]] += 1
        else:
            rocket_dict[launch["rocket"]] = 1
    for key, value in sorted(rocket_dict.items(),
                             key=lambda kv: kv[1], reverse=True):
        rurl = "https://api.spacexdata.com/v4/rockets/" + key
        req = requests.get(rurl)

        print(req.json()["name"] + ": " + str(value))
=======
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>
ordered by the number of launches in descending order or,
if rockets have the same amount of launches, in alphabetical order
"""


import requests


if __name__ == "__main__":
    url = 'https://api.spacexdata.com/v4/launches'
    results = requests.get(url).json()
    rocketDict = {}
    for launch in results:
        rocket = launch.get('rocket')
        url = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
        results = requests.get(url).json()
        rocket = results.get('name')
        if rocketDict.get(rocket) is None:
            rocketDict[rocket] = 1
        else:
            rocketDict[rocket] += 1
    rocketList = sorted(rocketDict.items(), key=lambda kv: kv[0])
    rocketList = sorted(rocketList, key=lambda kv: kv[1], reverse=True)
    for rocket in rocketList:
        print("{}: {}".format(rocket[0], rocket[1]))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
