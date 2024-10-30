from pymongo import MongoClient
import mongo_database as db
import datetime

# author lists
OWR_LIST = []
OTOR_LIST = []
RTOO_LIST = []

with open('9-10_parser_code/author_categories/OWR_authors.txt', "r") as file:
    for line in file:
        OWR_LIST.append(line.strip())
with open('9-10_parser_code/author_categories/OtoR_authors.txt', "r") as file:
    for line in file:
        OTOR_LIST.append(line.strip())
with open('9-10_parser_code/author_categories/RtoO_authors.txt', "r") as file:
    for line in file:
        RTOO_LIST.append(line.strip())

IGNORED_USERS = db.ignored_users

def list_average_authors(collection, user_list):
    pipeline = [
        {"$match": {
            "author": {"$in": user_list},
            "selftext": {"$nin": ["", "[removed]"]}
        }},
        {"$group": {
            "_id": "$author",
            "count": {"$sum": 1}
        }},
        {"$match": {
            "count": {"$gte": 6, "$lte": 10}
        }}
    ]
    return [obj['_id'] for  obj in list(collection.aggregate(pipeline))]

def print_authors_posts_in_order(collection, author):
    pipeline = [
        {"$match": {
            "author": author
        }},
        
        {"$sort": {
            "created_utc": 1
        }}
    ]
    cursor = collection.aggregate(pipeline)
    for doc in cursor:
        print(f"SUBREDDIT: {doc['subreddit']}, TIME: {datetime.datetime.fromtimestamp(doc['created_utc'])}, TITLE: {doc['title']}, POST BODY: {doc['selftext']}\n\n\n\n")

def print_posts_in_subreddit(collection, subreddit, sample_size=5):
    pipeline = [
        {"$match": {
            "subreddit": subreddit,
            "selftext": {"$nin": ["", "[removed]"]}, 
            "author": {"$nin": IGNORED_USERS}
        }}, 
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    for doc in cursor:
        print(doc)

if __name__ == "__main__":
    client = MongoClient()
    db = client['data']
    collection = db['author_submissions']
    # print(list_average_authors(collection, OTOR_LIST))
    # print_authors_posts_in_order(collection, 'hebruiser79')
# ['ophelia0103', 'hannahfitched12', 'GratefulDanny', 'DryMouthMonster', 'hebruiser79']