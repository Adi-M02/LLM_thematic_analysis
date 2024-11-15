from pymongo import MongoClient
import csv
import os

OR_SUBREDDITS = ['opiates', 'opiatechurch', 'poppytea', 'heroin', 'glassine', 'opiatescirclejerk', 'heroinhighway', 'opiatesrecovery', 'suboxone', 'methadone', 'addiction', 'opiatewithdrawal', 'redditorsinrecovery', 'narcoticsanonymous', 'recovery', 'subutex', 'naranon', 'smartrecovery', 'buddhistrecovery', 'drugrehabcenters']


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


def write_user_upvotes(collection):
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
    write_dict_to_csv(user_upvotes, "influence/OR_user_upvotes.csv", ["author", "upvotes"])


if __name__ == "__main__":
    client = MongoClient()
    db = client['reddit']
    collection = db['posts_only']
    write_user_upvotes(collection)