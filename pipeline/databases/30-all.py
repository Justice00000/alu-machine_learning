#!/usr/bin/env python3
"""
<<<<<<< HEAD
List all documents
=======
Defines function that lists all documents in MongoDB collection
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


def list_all(mongo_collection):
<<<<<<< HEAD
    """ Return a list of all documents

    Args:
        mongo_collection (mongocollection): Mongo collection

    Returns:
        _type_: _description_
    """
    all_documents = mongo_collection.find()
    documents_list = list(all_documents)

    return documents_list
=======
    """
    Lists all documents in given MongoDB collection

    parameters:
        mongo_collection: the collection to use

    returns:
        list of all documents or 0 if no documents found
    """
    all_docs = []
    collection = mongo_collection.find()
    for document in collection:
        all_docs.append(document)
    return all_docs
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
