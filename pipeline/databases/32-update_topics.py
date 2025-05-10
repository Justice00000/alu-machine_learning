#!/usr/bin/env python3
"""
<<<<<<< HEAD
Update attributes
=======
Defines function that changes all topics of a school document based on name
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


def update_topics(mongo_collection, name, topics):
<<<<<<< HEAD
    """ Changes all topics of a school document based on the name

    Args:
        mongo_collection (object): pymongo collection object
        name (string): school name to update
        topics (list): list of topics approached in the school
    """
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})
=======
    """
    Changes all topics of a school document based on the name

    parameters:
        mongo_collection [pymongo]:
            the MongoDB collection to use
        name [string]:
            the school name to update
        topics [list of strings]:
            list of topics approached in the school
    """
    mongo_collection.update_many({'name': name},
                                 {'$set': {'topics': topics}})
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
