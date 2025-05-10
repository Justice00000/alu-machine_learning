#!/usr/bin/env python3
<<<<<<< HEAD
"""Pipeline Api"""
import requests
from datetime import datetime


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    recent = 0

    for dic in r.json():
        new = int(dic["date_unix"])
        if recent == 0 or new < recent:
            recent = new
            launch_name = dic["name"]
            date = dic["date_local"]
            rocket_number = dic["rocket"]
            launch_number = dic["launchpad"]

    rurl = "https://api.spacexdata.com/v4/rockets/" + rocket_number
    rocket_name = requests.get(rurl).json()["name"]
    lurl = "https://api.spacexdata.com/v4/launchpads/" + launch_number
    launchpad = requests.get(lurl)
    launchpad_name = launchpad.json()["name"]
    launchpad_local = launchpad.json()["locality"]
    string = "{} ({}) {} - {} ({})".format(launch_name, date, rocket_name,
                                           launchpad_name, launchpad_local)

    print(string)
=======
"""
Uses the (unofficial) SpaceX API to print the upcoming launch as:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

The “upcoming launch” is the one which is the soonest from now, in UTC
and if 2 launches have the same date, it's the first one in the API result.
"""


import requests


if __name__ == "__main__":
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    results = requests.get(url).json()
    dateCheck = float('inf')
    launchName = None
    rocket = None
    launchPad = None
    location = None
    for launch in results:
        launchDate = launch.get('date_unix')
        if launchDate < dateCheck:
            dateCheck = launchDate
            date = launch.get('date_local')
            launchName = launch.get('name')
            rocket = launch.get('rocket')
            launchPad = launch.get('launchpad')
    if rocket:
        rocket = requests.get('https://api.spacexdata.com/v4/rockets/{}'.
                              format(rocket)).json().get('name')
    if launchPad:
        launchpad = requests.get('https://api.spacexdata.com/v4/launchpads/{}'.
                                 format(launchPad)).json()
        launchPad = launchpad.get('name')
        location = launchpad.get('locality')

    print("{} ({}) {} - {} ({})".format(
        launchName, date, rocket, launchPad, location))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
