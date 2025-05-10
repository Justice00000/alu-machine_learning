#!/usr/bin/env python3
"""
<<<<<<< HEAD
Update attributes
=======
Defines function that returns the list of schools with a specific topic
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


def schools_by_topic(mongo_collection, topic):
<<<<<<< HEAD
    """ returns the lsit of schools having a specific topic

    Args:
        mongo_collection (object): pymongo collection object
        topic (string): topic searched
        
    """
    schools = mongo_collection.find({"topics": topic})
    return schools
=======
    """
    Finds list of all schools with a specific topic

    parameters:
        mongo_collection [pymongo]:
            the MongoDB collection to use
        topic [string]:
            the topic to search for

    returns:
        list of schools with the given topic
    """
    schools = []
    documents = mongo_collection.find({'topics': {'$all': [topic]}})
    for doc in documents:
        schools.append(doc)
    return schools
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
