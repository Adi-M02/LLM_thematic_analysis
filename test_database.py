from pymongo import MongoClient
import mongo_database as db


AUTHORS = set()
with open('data/authors.txt', 'r') as f:
    for line in f:
        AUTHORS.add(line.strip())
BOTS = db.ignored_users

def list_databases():
    client = MongoClient()
    print(client.list_database_names())

def list_collections():
    client = MongoClient()
    db = client['reddit']
    print(db.list_collection_names())

def estimate_size(collection):
    print(collection.estimatedDocumentCount())

def count_posts(collection):
    num_entries = collection.count_documents({
        "is_post": True,
    })
    print(num_entries)

def count_comments(collection):
    num_entries = collection.count_documents({
        "is_post": False,
    })
    print(num_entries)

def find_duplicates(collection):
    pipeline = [
        {"$match": {
            "selftext": {"$exists": True}
        }},
        {"$group": {
            "_id": {
                "created_utc": "$created_utc",
                "author": "$author"
            },
            "count": {"$sum": 1},
            "docs": {"$push": "$$ROOT"}
        }},
        {"$match": {
            "count": {"$gt": 1}
        }}
    ]
    cursor = collection.aggregate(pipeline)
    duplicates = []
    for doc in cursor:
        # print(f"Duplicate created_utc: {doc['_id']}")
        for post in doc['docs']:
            duplicates.append(post)
        #     print(post)
    print(len(duplicates))

def count_user_entries(collection, user_list):
    pipeline = [
        {"$match": {"author": {"$in": user_list}}},
        {"$count": "total_posts"}
    ]

    # Execute the aggregation pipeline
    result = list(collection.aggregate(pipeline))
    print(result)

def delete_user_comments(collection, author_list, batch_size):
    cursor = collection.find({
        "author": {"$in": author_list},       # Match if author is in the list
        "selftext": {"$exists": False}        # Only match if selftext doesn't exist
    }, {"_id": 1}).batch_size(batch_size)     # Only fetch the _id field to save memory

    # Process and delete documents in batches
    deleted_count = 0
    for document in cursor:
        # Delete the document by its _id
        collection.delete_one({"_id": document["_id"]})
        deleted_count += 1

        # Print progress every 1000 deletions
        if deleted_count % batch_size == 0:
            print(f"Deleted {deleted_count} documents.")

def delete_user_posts(collection, author_list, batch_size):
    cursor = collection.find({
        "author": {"$in": author_list},       # Match if author is in the list
        "selftext": {"$exists": True}        # Only match if selftext doesn't exist
    }, {"_id": 1}).batch_size(batch_size)     # Only fetch the _id field to save memory
    # Process and delete documents in batches
    deleted_count = 0
    for document in cursor:
        # Delete the document by its _id
        collection.delete_one({"_id": document["_id"]})
        deleted_count += 1
        # Print progress every 1000 deletions
        if deleted_count % batch_size == 0:
            print(f"Deleted {deleted_count} documents.")

def count_commenters(collection):
    pipeline = [
        {"$match": {"is_post": False}},
        {"$group": {"_id": "$author"}},
        {"$count": "unique_authors"}
    ]
    result = collection.aggregate(pipeline)
    for doc in result:
        print(f"Number of unique authors where is_post is False: {doc['unique_authors']}")

def get_posts_in_subreddits(collection, subreddit_list):
    pipeline = [
        {
            "$match": {
                "is_post": True,
                "subreddit": {"$in": subreddit_list},
                "created_utc": {
                    "$gte": 1609459200,  # January 1, 2021
                    "$lte": 1672444799   # December 31, 2022 (end of day)
                }
            }
        }
    ]
    cursor = collection.aggregate(pipeline)
    return cursor

if __name__ == "__main__":
    client = MongoClient()
    db = client['reddit']
    collection = db['posts_and_comments']
    get_posts_in_subreddits(collection, ["opiatesrecovery", "opiates"])


    client.close()