#!/usr/bin/env python3
<<<<<<< HEAD
"""Pipeline Api"""
=======
"""
Defines methods to ping the Star Wars API and return the list of home planets
for all sentient species
"""


>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import requests


def sentientPlanets():
<<<<<<< HEAD
    """returns the list of names of the home planets of all sentient species"""
    url = "https://swapi-api.hbtn.io/api/species"
    r = requests.get(url)
    world_list = []
    while r.status_code == 200:
        for species in r.json()["results"]:
            url = species["homeworld"]
            if url is not None:
                ur = requests.get(url)
                world_list.append(ur.json()["name"])
        try:
            r = requests.get(r.json()["next"])
        except Exception:
            break
    return world_list
=======
    """
    Uses the Star Wars API to return the list of home planets
        for all sentient species

    returns:
        [list]: home planets of sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    speciesList = []
    while url:
        results = requests.get(url).json()
        speciesList += results.get('results')
        url = results.get('next')
    homePlanets = []
    for species in speciesList:
        if species.get('designation') == 'sentient' or \
           species.get('classification') == 'sentient':
            url = species.get('homeworld')
            if url:
                planet = requests.get(url).json()
                homePlanets.append(planet.get('name'))
    return homePlanets
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
