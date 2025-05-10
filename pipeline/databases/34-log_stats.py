#!/usr/bin/env python3
<<<<<<< HEAD
"""
Nginx logs stored in MongoDB:
"""


from pymongo import MongoClient


if __name__ == '__main__':

    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    # Get the total number of documents
    total_logs = collection.count_documents({})

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = {method: collection.count_documents(
        {"method": method}) for method in methods}

    # Get the count of status check
    status_check_count = collection.count_documents(
        {"method": "GET", "path": "/status"})

    # Print the stats
    print(f"{total_logs} logs")
    print("Methods:")
    for method, count in method_counts.items():
        print(f"\tmethod {method}: {count}")
    print(f"{status_check_count} status check")
=======
"""Provide some stats about Nginx logs stored in MongoDB."""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_coll = client.logs.nginx
    doc_count = logs_coll.count_documents({})
    print("{} logs".format(doc_count))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = logs_coll.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, method_count))
    filter_path = {"method": "GET", "path": "/status"}
    path_count = logs_coll.count_documents(filter_path)
    print("{} status check".format(path_count))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
