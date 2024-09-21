# from insight import Insight
from insight_no_spellcheck import Insight
import mongo_database as db
import parse_using_dui as dui

from pymongo import MongoClient
import csv
import time
import os
# import matplotlib.pyplot as plt
# import addcopyfighandler

# author lists
OWR_list = []
OtoR_list = []
RtoO_list = []
with open('9-10_parser_code/author_categories/OWR_authors.txt', "r") as file:
    for line in file:
        OWR_list.append(line.strip())
with open('9-10_parser_code/author_categories/OtoR_authors.txt', "r") as file:
    for line in file:
        OtoR_list.append(line.strip())
with open('9-10_parser_code/author_categories/RtoO_authors.txt', "r") as file:
    for line in file:
        RtoO_list.append(line.strip())

# subreddit lists
opiate_subreddits = ['opiates', 'OpiateChurch', 'heroin', 'PoppyTea', 'glassine', 'opiatescirclejerk', 'HeroinHighway']
opiate_recovery_subreddits = ['OpiatesRecovery', 'suboxone', 'Methadone', 'addiction', 'REDDITORSINRECOVERY', 'Opiatewithdrawal', 'recovery', 'NarcoticsAnonymous', 'Subutex', 'naranon', 'SMARTRecovery', 'buddhistrecovery', 'drugrehabcenters']

def write_to_csv_sorted(term_count_dict, output_file):
    term_count_dict = dict(sorted(term_count_dict.items(), key=lambda x: x[1], reverse=True))
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        open(output_file, 'w').close()
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['', 'Series1'])
        for key, value in term_count_dict.items():
            writer.writerow({'': key, 'Series1': value})

def write_top_20_to_csv(categories_frequency_dict, output_file):
    categories_frequency_dict = dict(sorted(categories_frequency_dict.items(), key=lambda x: x[1], reverse=True)[:20])
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        open(output_file, 'w').close()
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Category', 'Frequency'])
        writer.writeheader()
        for key, value in categories_frequency_dict.items():
            writer.writerow({'Category': key, 'Frequency': value}) 

def analyze_sample_from_user_list(client, user_list, sample_size, figure_name):
    posts = db.sample_posts_in_user_list_filtered(client, user_list, sample_size)
    start_time = time.time()
    analyzer = Insight()
    term_categories = dui.build_term_categories_dict('categories.csv')
    category_count = dui.build_category_count_dict('categories.csv')
    term_count = {}
    total_terms = 0
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count, num_terms, term_count = dui.add_counts_to_category_count_dict(analyzer, post, term_categories, category_count, term_count)
        total_terms += num_terms
    write_to_csv_sorted(category_count, f'8-9-analysis_results/{figure_name}/{figure_name}_category_data.csv')
    write_to_csv_sorted(term_count, f'8-9-analysis_results/{figure_name}/{figure_name}_term_data.csv')
    category_frequencies = dui.category_count_to_frequency_among_total_terms(category_count, total_terms)
    write_top_20_to_csv(category_frequencies, f'8-9-analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def analyze_sample_from_user_list_include_list(client, user_list, sample_size, figure_name, subreddit_list):
    posts = db.sample_user_list_in_subreddit_list_filtered(client, user_list, sample_size, subreddit_list)
    start_time = time.time()
    analyzer = Insight()
    term_categories = dui.build_term_categories_dict('categories.csv')
    category_count = dui.build_category_count_dict('categories.csv')
    term_count = {}
    total_terms = 0
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count, num_terms, term_count = dui.add_counts_to_category_count_dict(analyzer, post, term_categories, category_count, term_count)
        total_terms += num_terms
    write_to_csv_sorted(category_count, f'8-9-analysis_results/{figure_name}/{figure_name}_category_data.csv')
    write_to_csv_sorted(term_count, f'8-9-analysis_results/{figure_name}/{figure_name}_term_data.csv')
    category_frequencies = dui.category_count_to_frequency_among_total_terms(category_count, total_terms)
    write_top_20_to_csv(category_frequencies, f'8-9-analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def analyze_presence_absence(client, user_list, sample_size, figure_name):
    posts = db.sample_posts_in_user_list_filtered(client, user_list, sample_size)
    start_time = time.time()
    analyzer = Insight()
    term_categories = dui.build_term_categories_dict('categories.csv')
    category_count = dui.build_category_count_dict('categories.csv')
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count = dui.get_presence_of_categories(analyzer, post, term_categories, category_count)
    write_to_csv_sorted(category_count, f'8-9-analysis_results/{figure_name}/{figure_name}_category_data.csv')
    category_frequencies = {category: count/sample_size for category, count in category_count.items()}
    write_top_20_to_csv(category_frequencies, f'8-9-analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def analyze_presence_absence_include_list(client, user_list, sample_size, figure_name, subreddit_list):
    posts = db.sample_user_list_in_subreddit_list_filtered(client, user_list, sample_size, subreddit_list)
    start_time = time.time()
    analyzer = Insight()
    term_categories = dui.build_term_categories_dict('categories.csv')
    category_count = dui.build_category_count_dict('categories.csv')
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count = dui.get_presence_of_categories(analyzer, post, term_categories, category_count)
    write_to_csv_sorted(category_count, f'8-9-analysis_results/{figure_name}/{figure_name}_category_data.csv')
    category_frequencies = {category: count/sample_size for category, count in category_count.items()}
    write_top_20_to_csv(category_frequencies, f'8-9-analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

if __name__ == '__main__':
    client = MongoClient()
    # analyze_sample_from_user_list_include_list(client, OWR_list, 1000, 'figure_28_testing', opiate_subreddits)
    # analyze_sample_from_user_list(client, RtoO_list, 1000, 'figure_23_testing')
    # analyze_sample_from_user_list(client, OWR_list, 10000, 'figure_14')
    # analyze_sample_from_user_list(client, OtoR_list, 10000, 'figure_19')
    # analyze_sample_from_user_list(client, RtoO_list, 10000, 'figure_23')
    # analyze_sample_from_user_list_include_list(client, OWR_list, 10000, 'figure_28', opiate_subreddits)
    # analyze_sample_from_user_list_include_list(client, OtoR_list, 10000, 'figure_33', opiate_subreddits)
    # analyze_sample_from_user_list_include_list(client, RtoO_list, 10000, 'figure_38', opiate_subreddits)
    # analyze_sample_from_user_list_include_list(client, OtoR_list, 10000, 'figure_43', opiate_recovery_subreddits)
    # analyze_sample_from_user_list_include_list(client, RtoO_list, 10000, 'figure_48', opiate_recovery_subreddits)
    # analyze_presence_absence(client, OWR_list, 10000, 'figure_53')
    # analyze_presence_absence(client, OtoR_list, 10000, 'figure_54')
    # analyze_presence_absence(client, RtoO_list, 10000, 'figure_55')
    # analyze_presence_absence_include_list(client, OWR_list, 10000, 'figure_56', opiate_subreddits)
    # analyze_presence_absence_include_list(client, OtoR_list, 10000, 'figure_57', opiate_subreddits)
    # analyze_presence_absence_include_list(client, RtoO_list, 10000, 'figure_58', opiate_subreddits)
    # analyze_presence_absence_include_list(client, OtoR_list, 10000, 'figure_59', opiate_recovery_subreddits)
    # analyze_presence_absence_include_list(client, RtoO_list, 10000, 'figure_60', opiate_recovery_subreddits)

    # dui.plot_four_levels('8-9-analysis_results/figure_48/figure_48_term_data.csv')