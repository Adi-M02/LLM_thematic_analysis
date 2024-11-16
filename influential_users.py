from pymongo import MongoClient
import csv
import os

OR_SUBREDDITS = ['opiates', 'opiatechurch', 'poppytea', 'heroin', 'glassine', 'opiatescirclejerk', 'heroinhighway', 'opiatesrecovery', 'suboxone', 'methadone', 'addiction', 'opiatewithdrawal', 'redditorsinrecovery', 'narcoticsanonymous', 'recovery', 'subutex', 'naranon', 'smartrecovery', 'buddhistrecovery', 'drugrehabcenters']
OPIATE_SUBREDDITS = ['opiates', 'opiatechurch', 'heroin', 'poppytea', 'glassine', 'opiatescirclejerk', 'heroinhighway']
OPIATE_RECOVERY_SUBREDDITS = ['opiatesrecovery', 'suboxone', 'methadone', 'addiction', 'redditorsinrecovery', 'opiatewithdrawal', 'recovery', 'narcoticsanonymous', 'subutex', 'naranon', 'smartrecovery', 'buddhistrecovery', 'drugrehabcenters']


def write_dict_to_csv(data, output_path, headers):
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for key, value in data.items():
            writer.writerow([key, value])


def write_user_upvotes(collection, output):
    pipeline = [
        {
            "$match": {
                "subreddit": {"$in": OR_SUBREDDITS}
            }
        },
        {
            "$group": {
                "_id": "$author",
                "total_score": {"$sum": "$score"}
            }
        }
    ]

    # Execute the pipeline
    user_upvotes = {doc["_id"]: doc["total_score"] for doc in collection.aggregate(pipeline)}
    write_dict_to_csv(user_upvotes, output, ["author", "upvotes"])

def write_user_upvotes_by_subreddit(collection, subreddit, output):
    pipeline = [
        {
            "$match": {
                "subreddit": subreddit
            }
        },
        {
            "$group": {
                "_id": "$author",
                "total_score": {"$sum": "$score"}
            }
        }
    ]

    # Execute the pipeline
    user_upvotes = {doc["_id"]: doc["total_score"] for doc in collection.aggregate(pipeline)}
    user_upvotes = dict(sorted(user_upvotes.items(), key=lambda item: item[1], reverse=True))
    write_dict_to_csv(user_upvotes, output, ["author", "upvotes"])


if __name__ == "__main__":
    client = MongoClient()
    db = client['reddit']
    collection = db['posts_only']
    for subreddit in OPIATE_SUBREDDITS:
        write_user_upvotes_by_subreddit(collection, subreddit, f"influence/user_upvotes_OR_subreddits/opiate_subreddits/{subreddit}.csv")
