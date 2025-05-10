#!/usr/bin/env python3
"""
<<<<<<< HEAD
Insert document
=======
Defines function that inserts a new document in a MongoDB collection
   based on kwargs
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


def insert_school(mongo_collection, **kwargs):
<<<<<<< HEAD
    """ Insert document in a collection based on kwargs

    Args:
        mongo_collection (_type_): _description_
    Returns: new _id
    """
    new_id = mongo_collection.insert_one(kwargs).inserted_id
    return new_id
=======
    """
    Inserts a new document in a MongoDB collection based on kwargs

    parameters:
        kwargs: the new document to add

    returns:
        the new _id
    """
    document = mongo_collection.insert_one(kwargs)
    return (document.inserted_id)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
