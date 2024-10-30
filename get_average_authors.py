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

OR_SUBREDDITS = ['opiates', 'opiatechurch', 'poppytea', 'heroin', 'glassine', 'opiatescirclejerk', 'heroinhighway', 'opiatesrecovery', 'suboxone', 'methadone', 'addiction', 'opiatewithdrawal', 'redditorsinrecovery', 'narcoticsanonymous', 'recovery', 'subutex', 'naranon', 'smartrecovery', 'buddhistrecovery', 'drugrehabcenters']

IGNORED_USERS = db.ignored_users

AUTHORS = set()
with open('data/authors.txt', 'r') as f:
    for line in f:
        AUTHORS.add(line.strip())


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

def write_authors_posts_in_order(collection, author):
    pipeline = [
        {"$match": {
            "author": author
        }},
        
        {"$sort": {
            "created_utc": 1
        }}
    ]
    with open(f"author_post_histories/{author}.txt", 'w') as f:
        cursor = collection.aggregate(pipeline)
        for doc in cursor:
            print(doc)
            if doc['is_post']:
                output = (
                    f"POST:\n"
                    f"SUBREDDIT: {doc['subreddit']}, TIME: {datetime.datetime.fromtimestamp(doc['created_utc'])}, "
                    f"TITLE: {doc['title']}, POST BODY: {doc['selftext']}, PERMALINK: {doc['permalink']}\n\n"
                )
            else:
                output = (
                    f"COMMENT:\n"
                    f"SUBREDDIT: {doc['subreddit']}, TIME: {datetime.datetime.fromtimestamp(doc['created_utc'])}, "
                    f"BODY: {doc['body']}\n\n"
                )      
            # f.write(output)          


# def print_author_post_and_comment_chain_in_order(collection, author):
#     pipeline = [
#         {"$match": {
#             "author": author
#         }},
        
#         {"$sort": {
#             "created_utc": 1
#         }}
#     ]
#     cursor = collection.aggregate(pipeline)
#     for doc in cursor:

#         if doc['is_post']:
#             print(f"POST:\nSUBREDDIT: {doc['subreddit']}, TIME: {datetime.datetime.fromtimestamp(doc['created_utc'])}, TITLE: {doc['title']}, POST BODY: {doc['selftext']}, URL: {doc['url']}\n\n\n\n")
#         else:
            
#             print(f"COMMENT:\nSUBREDDIT: {doc['subreddit']}, TIME: {datetime.datetime.fromtimestamp(doc['created_utc'])}, BODY: {doc['body']}\n\n\n\n")    

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

def get_users_with_post_interval(collection, years):
    interval_seconds = years * 31536000
    pipeline = [
        {
            "$match": {
                "is_post": True  # Only consider entries where is_post is True
            }
        },
        {
            "$group": {
                "_id": "$author",  # Group by author field
                "first_post_time": {"$min": "$created_utc"},  # Earliest post time
                "last_post_time": {"$max": "$created_utc"},   # Latest post time
            }
        },
        {
            "$project": {
                "time_diff": {"$subtract": ["$last_post_time", "$first_post_time"]}  # Difference in seconds
            }
        },
        {
            "$match": {
                "time_diff": {"$gte": interval_seconds}
            }
        },   
        {
            "$project": {
                "_id": 1
            }
        } 
    ]
    result = collection.aggregate(pipeline)
    out = []
    for doc in result:
        out.append(doc['_id'])
    return out

def get_users_with_valid_posts(collection, user_list, threshold):
    valid_users = []
    for user in user_list:
        post_count = collection.count_documents({
            "author": user, 
            "is_post": True,
            "seftext": {"$nin": ["", "removed"]},
            "$expr": {
                "$in": [{"$toLower": "$subreddit"}, OR_SUBREDDITS]  # Convert subreddit in DB to lowercase
            }
        })
        if post_count >= threshold:
            valid_users.append(user)
    return valid_users

if __name__ == "__main__":
    client = MongoClient()
    db = client['data']
    collection = db['author_submissions']
    # print(list_average_authors(collection, OTOR_LIST))
    # print_authors_posts_in_order(collection, 'hebruiser79')
# ['ophelia0103', 'hannahfitched12', 'GratefulDanny', 'DryMouthMonster', 'hebruiser79']
