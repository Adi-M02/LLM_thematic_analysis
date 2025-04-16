import csv
import re
import html
coding_file = '/local/disk2/not_backed_up/amukundan/Thematic Analysis Opiate Subreddits/All_Codes_Manual_Analysis_fixEncoding.csv'

def parse_csv(coding_file):
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
            except:
                title = html.unescape(process_post_field(row['Post']))
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
            print("-" * 80)
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
    
def preprocess_post(result):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|&iquest;|&Ecirc;|&Euml;|&quot;|&Ouml;|&reg;|&uuml;|&ordf;|&sect;|&brvbar;|&iexcl;|&cent;|&acirc;|&micro;|&Agrave;|&pound;|&ugrave;|&divide;|&Uuml;|&ograve;|&frac14;|&yuml;|&raquo;|&cedil;|&euro;|&lt;|&egrave;|&plusmn;|&para;|&Egrave;|&Uacute;|&ouml;|&curren;|&Auml;|&yacute;|&Oslash;|&Ccedil;|&Icirc;|&auml;|&apos;|&oacute;|&ocirc;|&THORN;|&uml;|&Ucirc;|&Aacute;|&ecirc;|&not;|&ntilde;|&laquo;|&deg;|&thorn;|&AElig;|&aacute;|&uacute;|&Iacute;|&euml;|&Eacute;|&igrave;|&agrave;|&middot;|&Ograve;|&otilde;|&iuml;|&frac12;|&Igrave;|&Yacute;|&Atilde;|&aring;|&Oacute;|&macr;|&ordm;|&atilde;|&frac34;|&Acirc;|&sup3;|&nbsp;|&icirc;|&Iuml;|&eacute;|&oslash;|&acute;|&sup1;|&ucirc;|&copy;|&iacute;|&Aring;|&yen;|&Ugrave;|&ccedil;|&amp;|&Ocirc;|&shy;|&Otilde;|&eth;|&ETH;|&gt;|&sup2;|&times;|&szlig;|&Ntilde;|&aelig;'
    cleaned_post = re.sub(pattern, '&', result)
    cleaned_post = re.sub(pattern, ' ', cleaned_post)
    return cleaned_post    
if __name__ == "__main__":
    parse_csv(coding_file)