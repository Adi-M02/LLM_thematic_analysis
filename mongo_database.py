from pymongo import MongoClient
import re

# bot user list
ignored_users = ['AutoModerator', 'TrendingBot', 'nsfw_celbs', 'HotMomentumStocks']

#html entities
html_identities = ['&iquest;', '&Ecirc;', '&Euml;', '&quot;', '&Ouml;', '&reg;', '&uuml;', '&ordf;', '&sect;', '&brvbar;', '&iexcl;', '&cent;', '&acirc;', '&micro;', '&Agrave;', '&pound;', '&ugrave;', '&divide;', '&Uuml;', '&ograve;', '&frac14;', '&yuml;', '&raquo;', '&cedil;', '&euro;', '&lt;', '&egrave;', '&plusmn;', '&para;', '&Egrave;', '&Uacute;', '&ouml;', '&curren;', '&Auml;', '&yacute;', '&Oslash;', '&Ccedil;', '&Icirc;', '&auml;', '&apos;', '&oacute;', '&ocirc;', '&THORN;', '&uml;', '&Ucirc;', '&Aacute;', '&ecirc;', '&not;', '&ntilde;', '&laquo;', '&deg;', '&thorn;', '&AElig;', '&aacute;', '&uacute;', '&Iacute;', '&euml;', '&Eacute;', '&igrave;', '&agrave;', '&middot;', '&Ograve;', '&otilde;', '&iuml;', '&frac12;', '&Igrave;', '&Yacute;', '&Atilde;', '&aring;', '&Oacute;', '&macr;', '&ordm;', '&atilde;', '&frac34;', '&Acirc;', '&sup3;', '&nbsp;', '&icirc;', '&Iuml;', '&eacute;', '&oslash;', '&acute;', '&sup1;', '&ucirc;', '&copy;', '&iacute;', '&Aring;', '&yen;', '&Ugrave;', '&ccedil;', '&amp;', '&Ocirc;', '&shy;', '&Otilde;', '&eth;', '&ETH;', '&gt;', '&sup2;', '&times;', '&szlig;', '&Ntilde;', '&aelig;']

def get_raw_post_in_subreddits(client, subreddit_list, sample_size, ignored_users):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "subreddit": {"$in": subreddit_list},
            "selftext": {"$nin": ["", "[removed]"]}, # Exclude empty strings and "[removed]"
            "author": {"$nin": ignored_users}
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        results.append(post)
    return results

def sample_subreddits_preprocessed(client, subreddit_list, sample_size, ignored_users):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "subreddit": {"$in": subreddit_list},
            "selftext": {"$nin": ["", "[removed]"]}, # Exclude empty strings and "[removed]"
            "author": {"$nin": ignored_users}
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        results.append(post['selftext'])
    return preprocess_posts(results)
    

def get_random_sample(client, sample_size):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "selftext": {"$nin": ["", "[removed]"]}  # Exclude empty strings and "[removed]"
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        selftext = post['selftext']
        results.append(selftext)
    return results

def get_sample_in_subreddits(client, subreddit_list, sample_size):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "subreddit": {"$in": subreddit_list},
            "selftext": {"$nin": ["", "[removed]"]}  # Exclude empty strings and "[removed]"
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        selftext = post['selftext']
        results.append(selftext)
    return results

def get_sample_of_posts_in_user_list(client, user_list, sample_size):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "author": {"$in": user_list},
            "selftext": {"$nin": ["", "[removed]"]}  # Exclude empty strings and "[removed]"
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        selftext = post['selftext']
        results.append(selftext)
    return results

def get_sample_of_posts_in_user_list_in_subreddit_list(client, user_list, sample_size, subreddit_list):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "author": {"$in": user_list},
            "selftext": {"$nin": ["", "[removed]"]},  # Exclude empty strings and "[removed]"
            "subreddit": {"$in": subreddit_list}
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        selftext = post['selftext']
        results.append(selftext)
    return results

# pre processing of posts to remove links, special characters
def preprocess_posts(results):
    cleaned_results = []
    for post in results:
        new_post = preprocess_post(post)
        if new_post:
            cleaned_results.append(new_post)
    return cleaned_results

def preprocess_post(result):
    # pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|&amp;'
    # cleaned_post = re.sub(pattern, ' ', result)
    # return cleaned_post
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|&iquest;|&Ecirc;|&Euml;|&quot;|&Ouml;|&reg;|&uuml;|&ordf;|&sect;|&brvbar;|&iexcl;|&cent;|&acirc;|&micro;|&Agrave;|&pound;|&ugrave;|&divide;|&Uuml;|&ograve;|&frac14;|&yuml;|&raquo;|&cedil;|&euro;|&lt;|&egrave;|&plusmn;|&para;|&Egrave;|&Uacute;|&ouml;|&curren;|&Auml;|&yacute;|&Oslash;|&Ccedil;|&Icirc;|&auml;|&apos;|&oacute;|&ocirc;|&THORN;|&uml;|&Ucirc;|&Aacute;|&ecirc;|&not;|&ntilde;|&laquo;|&deg;|&thorn;|&AElig;|&aacute;|&uacute;|&Iacute;|&euml;|&Eacute;|&igrave;|&agrave;|&middot;|&Ograve;|&otilde;|&iuml;|&frac12;|&Igrave;|&Yacute;|&Atilde;|&aring;|&Oacute;|&macr;|&ordm;|&atilde;|&frac34;|&Acirc;|&sup3;|&nbsp;|&icirc;|&Iuml;|&eacute;|&oslash;|&acute;|&sup1;|&ucirc;|&copy;|&iacute;|&Aring;|&yen;|&Ugrave;|&ccedil;|&amp;|&Ocirc;|&shy;|&Otilde;|&eth;|&ETH;|&gt;|&sup2;|&times;|&szlig;|&Ntilde;|&aelig;'
    cleaned_post = re.sub(pattern, '&', result)
    cleaned_post = re.sub(pattern, ' ', cleaned_post)
    return cleaned_post

def test_preprocess_post(result):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|&iquest;|&Ecirc;|&Euml;|&quot;|&Ouml;|&reg;|&uuml;|&ordf;|&sect;|&brvbar;|&iexcl;|&cent;|&acirc;|&micro;|&Agrave;|&pound;|&ugrave;|&divide;|&Uuml;|&ograve;|&frac14;|&yuml;|&raquo;|&cedil;|&euro;|&lt;|&egrave;|&plusmn;|&para;|&Egrave;|&Uacute;|&ouml;|&curren;|&Auml;|&yacute;|&Oslash;|&Ccedil;|&Icirc;|&auml;|&apos;|&oacute;|&ocirc;|&THORN;|&uml;|&Ucirc;|&Aacute;|&ecirc;|&not;|&ntilde;|&laquo;|&deg;|&thorn;|&AElig;|&aacute;|&uacute;|&Iacute;|&euml;|&Eacute;|&igrave;|&agrave;|&middot;|&Ograve;|&otilde;|&iuml;|&frac12;|&Igrave;|&Yacute;|&Atilde;|&aring;|&Oacute;|&macr;|&ordm;|&atilde;|&frac34;|&Acirc;|&sup3;|&nbsp;|&icirc;|&Iuml;|&eacute;|&oslash;|&acute;|&sup1;|&ucirc;|&copy;|&iacute;|&Aring;|&yen;|&Ugrave;|&ccedil;|&amp;|&Ocirc;|&shy;|&Otilde;|&eth;|&ETH;|&gt;|&sup2;|&times;|&szlig;|&Ntilde;|&aelig;'
    cleaned_post = re.sub(pattern, '&', result)
    cleaned_post = re.sub(pattern, ' ', cleaned_post)
    return cleaned_post

def sample_posts_in_user_list_filtered(client, user_list, sample_size):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "author": {"$in": user_list},
            "author": {"$nin": ignored_users},
            "selftext": {"$nin": ["", "[removed]"]}  # Exclude empty strings and "[removed]"
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        selftext = post['selftext']
        results.append(selftext)
    return preprocess_posts(results)

def sample_user_list_in_subreddit_list_filtered(client, user_list, sample_size, subreddit_list):
    db = client['data']
    collection = db['author_submissions']
    pipeline = [
        {"$match": {
            "author": {"$in": user_list},
            "author": {"$nin": ignored_users},
            "selftext": {"$nin": ["", "[removed]"]},  # Exclude empty strings and "[removed]"
            "subreddit": {"$in": subreddit_list}
        }},
        {"$sample": {"size": sample_size}}
    ]
    cursor = collection.aggregate(pipeline)
    results = []
    for post in cursor:
        selftext = post['selftext']
        results.append(selftext)
    return preprocess_posts(results)

if __name__ == "__main__":
    pass