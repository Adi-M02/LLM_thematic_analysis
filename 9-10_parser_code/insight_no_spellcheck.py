import os
import csv
import logging
# import xmltodict
import zipfile
import json
from copy import deepcopy
from string import punctuation
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# from spellchecker import SpellChecker

# GLOBALS
# SCRIPT_DIR = os.path.abspath(__file__)
# PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# TERMS_FILE = "%s/lib/terms.csv" % PROJECT_DIR

categories = {}
reverse_categories = {}
category_paths = {}


def load_categories(filename, default=True):
    temp_categories = {}
    temp_reverse_categories = {}
    temp_category_paths = {}

    temp_categories['positive_emotions'] = []
    temp_categories['negative_emotions'] = []

    with open(filename, 'r', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        for row in reader:
            term = '{0!s}'.format(row['Term']).strip()

            category_path = '.'.join([x.strip() for x in [row['L1'], row['L2'], row['L3'], row['L4']] if x.strip()])
            category = category_path.split('.')[-1]
            temp_category_paths[category] = category_path

            classification = '{0!s}'.format(row['Classification']).strip()
            sentiment = '{0!s}'.format(row['Emotional Type']).strip()
            if sentiment == 'N':
                temp_categories['negative_emotions'].append(term)
            elif sentiment == 'P':
                temp_categories['positive_emotions'].append(term)

            if classification == 'E':
                try:
                    temp_categories[category].append(term)
                except KeyError:
                    temp_categories[category] = [term]

                try:
                    temp_reverse_categories[term].append(category)
                except KeyError:
                    temp_reverse_categories[term] = [category]

    # Clear out duplicates
    for category in temp_categories:
        temp_categories[category] = list(set(temp_categories[category]))

    for term in temp_reverse_categories:
        temp_reverse_categories[term] = list(set(temp_reverse_categories[term]))

    if default:
        global categories, reverse_categories, category_paths
        categories = temp_categories
        reverse_categories = temp_reverse_categories
        category_paths = temp_category_paths
    else:
        return (temp_categories, temp_reverse_categories, temp_category_paths)


class InsightText(object):
    def __init__(self, tokens, additional_categories=None):
        # spellcheck = SpellChecker(distance=1)
        # spellcheck.word_frequency.load_text_file('terms.txt')
        temp_categories = {key: [0, set()] for key in categories}
        temp_reverse_categories = deepcopy(reverse_categories)
        temp_category_paths = deepcopy(category_paths)
        term_counts = {}

        if additional_categories:
            secondary_categories, secondary_reverse_categories, secondary_category_paths = additional_categories
            for key in secondary_categories:
                if key not in temp_categories:
                    temp_categories[key] = [0, set()]

            temp_category_paths.update(secondary_category_paths)

            for term in secondary_reverse_categories:
                if term not in temp_reverse_categories:
                    temp_reverse_categories[term] = secondary_reverse_categories[term]
                else:
                    temp_reverse_categories[term] = list(set(temp_reverse_categories[term] + secondary_reverse_categories[term]))

        # Filter out OTHER annotated categories by default
        for token in tokens:
            # token = spellcheck.correction(token)
            if token in temp_reverse_categories:

                try:
                    term_counts[token] += 1
                except KeyError:
                    term_counts[token] = 1

                for category in temp_reverse_categories[token]:
                    temp_categories[category][0] += 1
                    temp_categories[category][1].add(token)

        self.category_paths = temp_category_paths
        self.categories = {key: value for key, value in temp_categories.items() if value[0] > 0}
        self.term_counts = term_counts
        self.total_term_count = len(tokens)
        # print(self.total_term_count)


class Insight(object):
    def __init__(self):
        self.sentiment_terms = {}
        self.terms = ''
        self.bigram_terms = ''

    # @staticmethod
    def generate_tokens(self, text):
        # Filter punctuation, sans '#' from given sentence. Then, return tokenized sentence
        translate_dict = {}
        for char in punctuation:
            if char == '#':
                continue

            translate_dict[ord(char)] = ' '

        text = text.translate(translate_dict)

        # NLTK will sometimes split 'wanna' or 'gonna' into 'gon na'. Remove the second 'na' if it wasn't there before.
        pure_tokens = {x.strip() for x in text.split(' ') if x.strip()}
        if 'na' not in pure_tokens:
            unigram_tokens = [x for x in word_tokenize(text.lower()) if x != 'na']
        else:
            unigram_tokens = [x for x in word_tokenize(text.lower())]
        self.terms = unigram_tokens
        ngram_tokens = zip(*[unigram_tokens[i:] for i in range(2)])
        bigram_tokens = (["_".join(ngram) for ngram in ngram_tokens])
        self.bigram_terms = bigram_tokens
        return list(set(bigram_tokens + unigram_tokens))

    def analyze_text(self, text, additional_categories=None):
        tokens = self.generate_tokens(text)
        return InsightText(tokens, additional_categories=additional_categories)

    def analyze_primary_class(self, text):
        dui_categories = self.analyze_text(text)
        try:
            category_counts = [(key, value[0]) for key, value in dui_categories.categories.items()]
            category_counts = sorted(category_counts, key=lambda x: x[1], reverse=True)
            top_categories = []
            top_count = category_counts[0][1]

            for entry in category_counts:
                if entry[1] == top_count:
                    top_categories.append(entry[0])
                else:
                    break

            return sorted(top_categories)
        except (KeyError, IndexError):
            return []


# def parse_xml(file_handle):
#     json_obj = xmltodict.parse(file_handle)
#     dui_collections = {}
#     for collection in json_obj['dui']['collection']:
#         username = collection['user']
#         dui_collections[username] = {'files': [], 'timestamped': False}

#         documents = collection['document']
#         for document in documents:
#             try:
#                 filename = document['filename']
#                 categories = document['category']
#                 if not isinstance(categories, list):
#                     categories = [categories]
#             except KeyError:
#                 continue

#             dui_categories = {}
#             dui_tokens = {}
#             dui_token_count = 0

#             for category in categories:
#                 category_name = category['name'].split('.')[-1]
#                 category_count = int(category['occurrences'])
#                 category_terms = category['terms']['token']
#                 if not isinstance(category_terms, list):
#                     category_terms = [category_terms]

#                 for term in category_terms:
#                     term_name = term['name']
#                     term_count = int(term['count'])
#                     dui_token_count += term_count
#                     if term_name not in dui_tokens:
#                         dui_tokens[term_name] = term_count
#                     else:
#                         dui_tokens[term_name] += term_count

#                 dui_categories[category_name] = [category_count, set([x['name'] for x in category_terms])]

#             analysis = InsightText([])
#             analysis.categories = dui_categories
#             analysis.term_counts = dui_tokens
#             analysis.total_term_count = dui_token_count

#             dui_collections[username]['files'].append({
#                 'filename': filename,
#                 'analysis': analysis
#             })
#     return dui_collections


def parse_csv(file_handle):
    users = {}
    reader = csv.DictReader(file_handle, delimiter='\t')
    for row in reader:
        if not row:
            continue

        batch = row['batch'].strip()
        filename = row['filename'].strip()
        count = int(row['count'])
        classification = row['classification'].strip()
        category = [x.strip() for x in [row['L1 category'], row['L2 category'], row['L3 category'], row['L4 category']] if x.strip()][-1]

        term = row['term'].strip()

        if batch not in users:
            users[batch] = {}

        if filename not in users[batch]:
            users[batch][filename] = {}

        if classification == 'category':
            users[batch][filename][category] = {'count': count}
        else:
            try:
                users[batch][filename][category][term] = count
            except KeyError:
                users[batch][filename][category] = {term: count}

    # Shove CSV into DUI structure as defined by the ZIP upload
    dui_collections = {}
    for username in users:
        dui_collections[username] = {'files': [], 'timestamped': False}
        for filename in users[username]:
            document = users[username][filename]

            dui_categories = {}
            dui_tokens = {}
            dui_token_count = 0

            for category in document:
                category_count = document[category]['count']
                category_terms = set()
                for term in document[category]:
                    if term == 'count':
                        continue

                    category_terms.add(term)
                    term_count = document[category][term]
                    dui_token_count += term_count

                    if term not in dui_tokens:
                        dui_tokens[term] = term_count
                    else:
                        dui_tokens[term] += term_count

                dui_categories[category] = [category_count, category_terms]

            analysis = InsightText([])
            analysis.categories = dui_categories
            analysis.term_counts = dui_tokens
            analysis.total_term_count = dui_token_count

            dui_collections[username]['files'].append({
                'filename': filename,
                'analysis': analysis
            })

    return dui_collections


# def parse_input(batch_file):
#     results = {'batch_name': batch_file.name, 'collections': {}}
#     filename = batch_file.filename
#     analyzer = Insight()

#     # Parse .zip upload
#     if batch_file.upload_format <= 3:
#         stream = zipfile.ZipFile(filename, 'r')
#         for file in stream.filelist:
#             file_contents = b''
#             logging.info("Reading '%s'..." % file.filename)
#             # Skip directories and __/hidden folders

#             if file.filename.startswith('__') or file.filename.endswith('/') \
#                     or 'ds_store' in filename.lower():
#                 continue

#             # Read file, and store it in memory
#             file_contents += stream.read(file.filename)
#             try:
#                 file_contents = file_contents.decode('utf-8')
#             except UnicodeDecodeError:
#                 file_contents = file_contents.decode('unicode_escape')

#             try:
#                 category_filename = '%s-category' % batch_file.filename
#                 additional_categories = load_categories(category_filename, default=False)
#             except Exception as error:
#                 logging.warning("Category file not found or could not be loaded. Using default categories.")
#                 additional_categories = None

#             if additional_categories:
#                 analyzed_results = analyzer.analyze_text(file_contents, additional_categories=additional_categories)
#             else:
#                 analyzed_results = analyzer.analyze_text(file_contents)

#             logging.info("Analyzed '%s'." % file.filename)

#             directory_name = file.filename.split('/')[0]
#             if directory_name not in results['collections']:
#                 results['collections'][directory_name] = {'files': [],
#                                                           'timestamped': batch_file.upload_format == 1 or batch_file.upload_format == 3}

#             results['collections'][directory_name]['files'].append({
#                 'filename': file.filename,
#                 'analysis': analyzed_results
#             })

#     # Parse .xml upload
#     elif batch_file.upload_format == 4:
#         with open(filename, 'rb') as stream:
#             results['collections'] = parse_xml(stream)

#     # Parse .csv upload
#     elif batch_file.upload_format == 5:
#         with open(filename, 'r') as stream:
#             results['collections'] = parse_csv(stream)

#     return results


# RUNNER
# print("loading categories")
load_categories('9-10_parser_code/terms.csv')


# def example(dict1, dict2):
#     for key in dict1:
#         if key not in dict2:
#             "Key not found in dict2"
#         elif dict1[key] != dict2[key]:
#             "They don't match"
#         else
            # "Do Something with the Matching Information"
#             "They match"
#     return "All keys match"
# if __name__ == "__main__":
#     print("Loading categories from '%s'..." % TERMS_FILE)
#     load_categories(TERMS_FILE)
#     print('test')