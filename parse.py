import csv
import re
import html
coding_file = '/Users/adimukundan/Downloads/Thematic Analysis Opiate Subreddits/All_Codes_Manual_Analysis_fixEncoding.csv'

def state_label_to_string(state_label):
    if state_label == 0:
        return "use"
    elif state_label == 1:
        return "withdrawal"
    elif state_label == 2:
        return "recovery"
    elif state_label == 4:
        return "unknown"
    
def parse_csv(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    posts_and_titles = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        # Use the csv.DictReader to read rows as dictionaries
        reader = csv.DictReader(file)
        
        # Loop through each row in the CSV file
        i = 0
        for row in reader:
            i+=1
            # Access each field by its column name
            user = row['User']
            subreddit = row['Subreddit']
            post_id = row['Post ID']
            date_time = row['Date/Time']
            empty = row['Empty']
            state_label = row['State Label']
            try:
                title, post = process_post_field(row['Post'])
                post = html.unescape(post)
                title = html.unescape(title)
                posts_and_titles.append((post_id, post, title, state_label_to_string(int(state_label)), row['incorrect days clean']))
            except:
                title = html.unescape(process_post_field(row['Post']))
                posts_and_titles.append((post_id, None, title, state_label_to_string(int(state_label)), row['incorrect days clean']))
            question = row['question']
            incorrect_days_clean = row['incorrect days clean']
            tense = row['tense']
            atypical_information = row['atypical information']
            special_cases = row['special cases']
            use = row['use']
            withdrawal = row['withdrawal']
            recovery = row['recovery']
            co_use = row['co-use']
            is_imputed = row['Is imputed']
            imputed = row['imputed']
            # Print the fields (optional, you can process them as needed)
            # print(f"User: {user}, Subreddit: {subreddit}, Post ID: {post_id}, Date/Time: {date_time}")
            # print(f"Empty: {empty}, State Label: {state_label}, Post: {post}")
            # print(f"Question: {question}, Incorrect Days Clean: {incorrect_days_clean}, Tense: {tense}")
            # print(f"Atypical Information: {atypical_information}, Special Cases: {special_cases}")
            # print(f"Use: {use}, Withdrawal: {withdrawal}, Recovery: {recovery}")
            # print(f"Co-Use: {co_use}, Is Imputed: {is_imputed}, Imputed: {imputed}")
            # print("-" * 80)
        return posts_and_titles

def process_post_field(post_field):
    try:
        # Check for both title and post
        if "title:" in post_field and "post:" in post_field:
            # Extract title and post content
            title_start = post_field.find("title:") + len("title:")
            title_end = post_field.find("post:")
            title = post_field[title_start:title_end].strip()
            post = post_field[title_end + len("post:"):].strip()
            return title, post
        
        # Check for title and comments
        elif "title:" in post_field and "comment:" in post_field:
            # Extract only the title content
            title_start = post_field.find("title:") + len("title:")
            title_end = post_field.find("comment:")
            title = post_field[title_start:title_end].strip()
            return title
        
        # If none of the conditions are met, return the raw post field
        else:
            return post_field.strip()
    except Exception as e:
        return f"Error processing post field: {str(e)}"

def parse_incorrext_days_clean(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    incorrect_days_clean = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            i+=1
            user = row['User']
            subreddit = row['Subreddit']
            post_id = row['Post ID']
            date_time = row['Date/Time']
            empty = row['Empty']
            state_label = row['State Label']
            question = row['question']
            incorrect_days_clean = row['incorrect days clean']
            tense = row['tense']
            atypical_information = row['atypical information']
            special_cases = row['special cases']
            use = row['use']
            withdrawal = row['withdrawal']
            recovery = row['recovery']
            co_use = row['co-use']
            is_imputed = row['Is imputed']
            imputed = row['imputed']
            try:
                title, post = process_post_field(row['Post'])
                post = html.unescape(post)
                title = html.unescape(title)
                incorrect_days_clean.append((post_id, post, title, state_label_to_string(int(state_label)), row['incorrect days clean']))
            except:
                title = html.unescape(process_post_field(row['Post']))
                incorrect_days_clean.append((post_id, None, title, state_label_to_string(int(state_label)), row['incorrect days clean']))

        return incorrect_days_clean
    
def parse_tense(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    tense_list = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            i+=1
            post_id = row['Post ID']
            state_label = row['State Label']
            tense = row['tense']
            tense = [int(num) for num in tense.split(',')] if tense else []
            if int(row['State Label']) == 4:
                continue
            try:
                title, post = process_post_field(row['Post'])
                post = html.unescape(post)
                title = html.unescape(title)
                tense_list.append((post_id, post, title, state_label_to_string(int(state_label)), tense))
            except:
                title = html.unescape(process_post_field(row['Post']))
                tense_list.append((post_id, None, title, state_label_to_string(int(state_label)), tense))

        return tense_list
    
def parse_feature_post_title_threshold(feature, coding_file='All_Codes_Manual_Analysis_fixEncoding.csv', skip_unknown=True, max_posts_length=1500):
    mapping = {"question": "question", "incorrect_days_clean": "incorrect days clean", "tense": "tense", "atypical_information": "atypical information", "special_cases": "special cases", "use": "use", "withdrawal": "withdrawal", "recovery": "recovery", "co-use": "co-use", "off-topic": "off-topic", "imputed": "imputed"}
    out = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            post_id = row['Post ID']
            state_label = row['State Label']
            feature_list = row[mapping[feature]]
            feature_list = [int(num) for num in feature_list.split(',')]
            if skip_unknown:
                if int(row['State Label']) == 4:
                    continue
            try:
                title, post = process_post_field(row['Post'])
                post = html.unescape(post)
                title = html.unescape(title)
                if len(post) > max_posts_length:
                    continue
                out.append((post_id, post, title, state_label_to_string(int(state_label)), feature_list))
            except:
                continue
        return out
    
def parse_all_features(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv', skip_unknown=True, max_posts_length=1500):
    features = ["atypical_information", "co-use", "off-topic", "question", "recovery", "special_cases", "tense", "use", "withdrawal"]
    mapping = {"question": "question", "incorrect_days_clean": "incorrect days clean", "tense": "tense", "atypical_information": "atypical information", "special_cases": "special cases", "use": "use", "withdrawal": "withdrawal", "recovery": "recovery", "co-use": "co-use", "off-topic": "off-topic", "imputed": "imputed"}
    out = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            post_id = row['Post ID']
            state_label = row['State Label']
            if skip_unknown:
                if int(row['State Label']) == 4:
                    continue
                try:
                    title, post = process_post_field(row['Post'])
                    post = html.unescape(post)
                    title = html.unescape(title)
                    if len(post) > max_posts_length:
                        continue
                    feature_dict = {}
                    for feature in features:
                        feature_list = row[mapping[feature]]
                        feature_list = [int(num) for num in feature_list.split(',')]
                        feature_dict[feature] = feature_list
                    text = post + " " + title
                    out.append((text, feature_dict))
                except:
                    continue
        return out
    
def parse_all_features_with_post_id(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv', skip_unknown=True, max_posts_length=1500):
    features = ["atypical_information", "co-use", "off-topic", "question", "recovery", "special_cases", "tense", "use", "withdrawal"]
    mapping = {"question": "question", "incorrect_days_clean": "incorrect days clean", "tense": "tense", "atypical_information": "atypical information", "special_cases": "special cases", "use": "use", "withdrawal": "withdrawal", "recovery": "recovery", "co-use": "co-use", "off-topic": "off-topic", "imputed": "imputed"}
    out = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            post_id = row['Post ID']
            state_label = row['State Label']
            if skip_unknown:
                if int(row['State Label']) == 4:
                    continue
                try:
                    title, post = process_post_field(row['Post'])
                    post = html.unescape(post)
                    title = html.unescape(title)
                    if len(post) > max_posts_length:
                        continue
                    feature_dict = {}
                    for feature in features:
                        feature_list = row[mapping[feature]]
                        feature_list = [int(num) for num in feature_list.split(',')]
                        feature_dict[feature] = feature_list
                    text = post + " " + title
                    out.append((post_id, text, feature_dict))
                except:
                    continue
        return out
    
def get_post_title_string(logger, post_id, coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Post ID'] == post_id:
                try:
                    post, title = process_post_field(row['Post'])
                    post = html.unescape(post)
                    title = html.unescape(title)
                    return str(post) + " " + str(title)
                except:
                    title = process_post_field(row['Post'])
                    title = html.unescape(title)
                    return str(title)
        logger.error(f"Post ID {post_id} not found in the coding file.")
        return " "
                
def word_count(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        i = 0
        words = ""
        logger = ""
        for row in reader:
            post_id = row['Post ID']
            words += get_post_title_string(logger, post_id)
    print(len(words))

def get_posts_and_titles_only(coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    posts_and_titles = []
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            post_content = row['Post']
            if "title:" in post_content and "post:" in post_content:
                # Extract title and post content
                title_start = post_content.find("title:") + len("title:")
                title_end = post_content.find("post:")
                title = post_content[title_start:title_end].strip()
                post = post_content[title_end + len("post:"):].strip()
                title = html.unescape(title)
                post = html.unescape(post)
                posts_and_titles.append((row["Post ID"], post, title))
    return posts_and_titles

def get_post_theme_presence(post_id, coding_file='All_Codes_Manual_Analysis_fixEncoding.csv'):
    with open(coding_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Post ID'] == post_id:
                themes = []
                columns = ['question', 'tense', 'atypical information', 'special cases', 'use', 'withdrawal', 'recovery', 'co-use', 'off-topic']
                for column in columns:
                    themes.append((column, row[column]))
    return themes

if __name__ == "__main__":
    word_count()
    # parse_all_features()