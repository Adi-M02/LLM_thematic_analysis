from insight_no_spellcheck import Insight
import mongo_database as db

from pymongo import MongoClient
import os
import csv
import time
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

def dui_analysis(analyzer, query):
    query = ' '.join([x.lower() for x in query.split(' ')])
    results = [{'analysis': analyzer.analyze_text(query)}]
    category_terms_dict = {}
    for key, value in results[0]['analysis'].categories.items():
        category_terms_dict[key] = value[1]
    return category_terms_dict

def get_dui_terms(analyzer, query):
    dui_dict = dui_analysis(analyzer, query)
    terms = []
    for key, value in dui_dict.items():
        terms.extend(value)
    return terms

def build_term_categories_dict(category_file):
    term_category_dict = {}
    with open(category_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            categories = [category for category in line[1:] if category != '']
            if line[0] in term_category_dict:
                term_category_dict[line[0]] += categories
            else:
                term_category_dict[line[0]] = categories
    for key, value in term_category_dict.items():
        term_category_dict[key] = list(set(value))
    return term_category_dict

def build_term_category_tier_dict(category_file, tier):
    term_category_dict = {}
    with open(category_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            categories = [category for category in line[1:] if category != '']
            if len(categories) >= tier:
                if line[0] in term_category_dict:
                    term_category_dict[line[0]] += [categories[tier-1]]
                else:
                    term_category_dict[line[0]] = [categories[tier-1]]
    for key, value in term_category_dict.items():
        term_category_dict[key] = list(set(value))
    return term_category_dict


def build_category_count_dict(category_file):
    category_count_dict = {}
    with open(category_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            categories = [category for category in line[1:] if category != '']
            for category in categories:
                category_count_dict[category] = 0
    return category_count_dict

def build_category_term_dict(category_file):
    category_term_dict = {}
    with open(category_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            categories = [category for category in line[1:] if category != '']
            for category in categories:
                if category in category_term_dict:
                    category_term_dict[category].append(line[0])
                else:
                    category_term_dict[category] = [line[0]]
    return category_term_dict

def write_to_csv_sorted(term_count_dict, output_file):
    term_count_dict = dict(sorted(term_count_dict.items(), key=lambda x: x[1], reverse=True))
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['', 'Series1'])
        for key, value in term_count_dict.items():
            writer.writerow({'': key, 'Series1': value})

def category_count_to_frequency_among_total_terms(categories_count_dict, total_terms):
    categories_frequency_dict = {}
    for key in categories_count_dict:
        categories_frequency_dict[key] = (categories_count_dict[key] / total_terms)*100
    return categories_frequency_dict

def write_top_20_to_csv(categories_frequency_dict, output_file):
    categories_frequency_dict = dict(sorted(categories_frequency_dict.items(), key=lambda x: x[1], reverse=True)[:20])
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Category', 'Frequency'])
        writer.writeheader()
        for key, value in categories_frequency_dict.items():
            writer.writerow({'Category': key, 'Frequency': value}) 

def add_counts_to_category_count_dict(analyzer, post, term_categories, category_count, term_count):
    post_terms = get_dui_terms(analyzer, post)
    num_terms = len(analyzer.terms)
    for term in post_terms:
        if term in term_count:
            if "_" in term:
                term_count[term] += analyzer.bigram_terms.count(term)
            else:
                term_count[term] += analyzer.terms.count(term)
        else:
            if "_" in term:
                term_count[term] = analyzer.bigram_terms.count(term)
            else:
                term_count[term] = analyzer.terms.count(term)
        if term in term_categories:
            for category in term_categories[term]:
                if "_" in term:
                    category_count[category] += analyzer.bigram_terms.count(term)
                else:
                    category_count[category] += analyzer.terms.count(term)
    return category_count, num_terms, term_count

def analyze_sample_from_user_list(client, user_list, sample_size, figure_name):
    posts = db.get_sample_of_posts_in_user_list(client, user_list, sample_size)
    start_time = time.time()
    analyzer = Insight()
    term_categories = build_term_categories_dict('categories.csv')
    category_count = build_category_count_dict('categories.csv')
    term_count = {}
    total_terms = 0
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count, num_terms, term_count = add_counts_to_category_count_dict(analyzer, post, term_categories, category_count, term_count)
        total_terms += num_terms
    write_to_csv_sorted(category_count, f'analysis_results/{figure_name}/{figure_name}_category_data.csv')
    write_to_csv_sorted(term_count, f'analysis_results/{figure_name}/{figure_name}_term_data.csv')
    category_frequencies = category_count_to_frequency_among_total_terms(category_count, total_terms)
    write_top_20_to_csv(category_frequencies, f'analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def analyze_sample_from_user_list_include_list(client, user_list, sample_size, figure_name, subreddit_list):
    posts = db.get_sample_of_posts_in_user_list_in_subreddit_list(client, user_list, sample_size, subreddit_list)
    start_time = time.time()
    analyzer = Insight()
    term_categories = build_term_categories_dict('categories.csv')
    category_count = build_category_count_dict('categories.csv')
    term_count = {}
    total_terms = 0
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count, num_terms, term_count = add_counts_to_category_count_dict(analyzer, post, term_categories, category_count, term_count)
        total_terms += num_terms
    write_to_csv_sorted(category_count, f'analysis_results/{figure_name}/{figure_name}_category_data.csv')
    write_to_csv_sorted(term_count, f'analysis_results/{figure_name}/{figure_name}_term_data.csv')
    category_frequencies = category_count_to_frequency_among_total_terms(category_count, total_terms)
    write_top_20_to_csv(category_frequencies, f'analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def get_presence_of_categories(analyzer, post, term_categories, categories_count_dict):
    post_terms = get_dui_terms(analyzer, post)
    dui_categories_in_post = set()
    for term in post_terms:
        if term in term_categories:
            dui_categories_in_post.update(set(term_categories[term]))
    for category in dui_categories_in_post:
        if category in categories_count_dict:
            categories_count_dict[category] += 1
        else:
            categories_count_dict[category] = 1
    return categories_count_dict

def analyze_presence_absence(client, user_list, sample_size, figure_name):
    posts = db.get_sample_of_posts_in_user_list(client, user_list, sample_size)
    start_time = time.time()
    analyzer = Insight()
    term_categories = build_term_categories_dict('categories.csv')
    category_count = build_category_count_dict('categories.csv')
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count = get_presence_of_categories(analyzer, post, term_categories, category_count)
    write_to_csv_sorted(category_count, f'analysis_results/{figure_name}/{figure_name}_category_data.csv')
    category_frequencies = {category: count/sample_size for category, count in category_count.items()}
    write_top_20_to_csv(category_frequencies, f'analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def analyze_presence_absence_include_list(client, user_list, sample_size, figure_name, subreddit_list):
    posts = db.get_sample_of_posts_in_user_list_in_subreddit_list(client, user_list, sample_size, subreddit_list)
    start_time = time.time()
    analyzer = Insight()
    term_categories = build_term_categories_dict('categories.csv')
    category_count = build_category_count_dict('categories.csv')
    i = 0
    for post in posts:
        i+=1
        if i % 1000 == 0:
            print(i, ' posts analyzed')
        category_count = get_presence_of_categories(analyzer, post, term_categories, category_count)
    write_to_csv_sorted(category_count, f'analysis_results/{figure_name}/{figure_name}_category_data.csv')
    category_frequencies = {category: count/sample_size for category, count in category_count.items()}
    write_top_20_to_csv(category_frequencies, f'analysis_results/{figure_name}/{figure_name}_top_20.csv')
    end_time = time.time()
    print('Time taken: ', (end_time - start_time)/60, ' minutes')

def get_tier_level_terms(category_file, tier_level):
    tier_terms = set()
    with open(category_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            categories = [category for category in line[1:] if category != '']
            if len(categories) >= tier_level:
                tier_terms.add(line[0])
    return tier_terms

def count_tier_level_categories(term_count_file, tier_terms):
    tier_level_counts = {}
    with open(term_count_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            if line[0] in tier_terms:
                tier_level_counts[line[0]] = int(line[1])
    return tier_level_counts
    
def plot_tier_one_category_counts(tier_one_counts, category_term):
    first_5 = {k: v for k, v in sorted(tier_one_counts.items(), key=lambda item: item[1], reverse=True)[:5]}
    print("Tier one:")
    for key in first_5:
        print(key)
    for key in first_5:
        print(first_5[key])
    terms = [term for term, count in tier_one_counts.items() if count > 0][::-1]
    counts = [count for term, count in tier_one_counts.items() if count > 0][::-1]
    sorted_counts = sorted(counts, reverse=True)
    # max_count = sorted_counts[1]
    max_count = sorted_counts[1]
    key_colors = {'drug_use': 'red', 'drug_terms': 'green', 'drug_recovery': 'yellow'}
    term_colors = {}
    for category in category_term:
        for term in category_term[category]:
            if term in terms:
                if category in key_colors:
                    term_colors[term] = key_colors[category]
    plt.figure(figsize=(10, 6))
    plt.bar(terms, counts, color=[term_colors[term] for term in terms])
    plt.xlabel('Term')
    plt.ylabel('Term Count')
    # plt.xticks([i * 1000 for i in range(len(terms) // 1000 + 1)], [str(i * 1000) for i in range(len(terms) // 1000 + 1)])
    plt.xticks([i * 1000 for i in range(len(terms) // 1000 + 1)], [str(i * 1000) for i in range(len(terms) // 1000 + 1)])
    # plt.yticks([i * 1000 for i in range(3)], [str(i * 1000) for i in range(3)])
    plt.yticks([i * 100 for i in range(max_count//100)], [str(i * 100) for i in range(max_count//100)])
    plt.ylim(0, max_count)
    legend_labels = {'drug_use': 'Drug Use', 'drug_terms': 'Drug Terms', 'drug_recovery': 'Drug Recovery'}
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in key_colors.values()]
    plt.legend(legend_handles, [legend_labels[key] for key in key_colors.keys()], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_tier_two_category_counts(tier_two_counts, category_term):
    first_5 = {k: v for k, v in sorted(tier_two_counts.items(), key=lambda item: item[1], reverse=True)[:5]}
    print("Tier two:")
    for key in first_5:
        print(key)
    for key in first_5:
        print(first_5[key])
    terms = [term for term, count in tier_two_counts.items() if count > 0][::-1]
    counts = [count for term, count in tier_two_counts.items() if count > 0][::-1]
    sorted_counts = sorted(counts, reverse=True)
    # max_count = sorted_counts[1]
    max_count = sorted_counts[1]
    key_colors = {'anxiolytics':'peachpuff', 'legal':'cornsilk', 'effects':'green', 'finance':'purple', 'drug_quantity':'olive', 'alcohol':'navy', 'cannabis':'deeppink', 'locations':'lightpink', 'acquisition':'plum', 'using':'firebrick', 'therapy':'cyan', 'chem_and_bio':'mediumspringgreen', 'stimulants':'orange', 'hallucinogens':'teal', 'addiction':'grey', 'opioids':'yellow', 'health':'blue', 'administration':'greenyellow', 'depressants':'goldenrod', 'withdrawal':'red'}
    term_colors = {}
    for category in category_term:
        for term in category_term[category]:
            if term in terms:
                if category in key_colors:
                    term_colors[term] = key_colors[category]
    plt.figure(figsize=(10, 6))
    plt.bar(terms, counts, color=[term_colors[term] for term in terms if term in term_colors])
    plt.xlabel('Term')
    plt.ylabel('Term Count')
    plt.xticks([i * 1000 for i in range(len(terms) // 1000 + 1)], [str(i * 1000) for i in range(len(terms) // 1000 + 1)])
    # plt.yticks([i * 1000 for i in range(3)], [str(i * 1000) for i in range(3)])
    plt.yticks([i * 100 for i in range(max_count//100-2)], [str(i * 100) for i in range(max_count//100-2)])
    plt.ylim(0, max_count)
    legend_labels = {'anxiolytics':'Anxiolytics', 'legal':'Legal', 'effects':'Effects', 'finance':'Finance', 'drug_quantity':'Drug Quantity', 'alcohol':'Alcohol', 'cannabis':'Cannabis', 'locations':'Locations', 'acquisition':'Acquisition', 'using':'Using', 'therapy':'Therapy', 'chem_and_bio':'Chemistry and Biology', 'stimulants':'Stimulants', 'hallucinogens':'Hallucinogens', 'addiction':'Addiction', 'opioids':'Opioids', 'health':'Health', 'administration':'Administration', 'depressants':'Depressants', 'withdrawal':'Withdrawal'}
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in key_colors.values()]
    plt.legend(legend_handles, [legend_labels[key] for key in key_colors.keys() if key], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_tier_three_category_counts(tier_three_counts, category_term):
    first_5 = {k: v for k, v in sorted(tier_three_counts.items(), key=lambda item: item[1], reverse=True)[:5]}
    print("Tier three:")
    for key in first_5:
        print(key)
    for key in first_5:
        print(first_5[key])
    terms = [term for term, count in tier_three_counts.items() if count > 0][::-1]
    counts = [count for term, count in tier_three_counts.items() if count > 0][::-1]
    sorted_counts = sorted(counts, reverse=True)
    # max_count = sorted_counts[1]
    max_count = sorted_counts[1]
    key_colors = {'health practioner':'magenta', 'mental disorder':'greenyellow', 'street':'teal', 'quitting':'yellow', 'euphoria':'pink', 'prescription':'cyan', 'neurological disorder':'cornsilk', 'oral':'violet', 'smoking':'blue', 'discomfort':'olive', 'drug':'green', 'physical':'purple', 'hospital department':'grey', 'withdrawal_symptoms':'red', 'nasal':'maroon', 'anxiety disorder':'mediumspringgreen', 'idu':'peachpuff', 'energetic':'navy'}
    term_colors = {}
    for category in category_term:
        for term in category_term[category]:
            if term in terms:
                if category in key_colors:
                    term_colors[term] = key_colors[category]
    plt.figure(figsize=(10, 6))
    plt.bar(terms, counts, color=[term_colors[term] for term in terms if term in term_colors])
    plt.xlabel('Term')
    plt.ylabel('Term Count')
    plt.xticks([i * 100 for i in range(len(terms) // 100 + 1)], [str(i * 100) for i in range(len(terms) // 100 + 1)])
    # plt.yticks([i * 1000 for i in range(3)], [str(i * 1000) for i in range(3)])
    plt.yticks([i * 100 for i in range(max_count//100)], [str(i * 100) for i in range(max_count//100)])
    plt.ylim(0, max_count)
    legend_labels = {'health practioner':'Health Practioner', 'mental disorder':'Mental Disorder', 'street':'Street', 'quitting':'Quitting', 'euphoria':'Euphoria', 'prescription':'Prescription', 'neurological disorder':'Neurological Disorder', 'oral':'Oral', 'smoking':'Smoking', 'discomfort':'Discomfort', 'drug':'Drug', 'physical':'Physical', 'hospital department':'Hospital Department', 'withdrawal_symptoms':'Withdrawal Symptoms', 'nasal':'Nasal', 'anxiety disorder':'Anxiety Disorder', 'idu':'IDU', 'energetic':'Energetic'}
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in key_colors.values()]
    plt.legend(legend_handles, [legend_labels[key] for key in key_colors.keys() if key], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_tier_four_category_counts(tier_four_counts, category_term):
    first_5 = {k: v for k, v in sorted(tier_four_counts.items(), key=lambda item: item[1], reverse=True)[:5]}
    print("Tier four:")
    for key in first_5:
        print(key)
    for key in first_5:
        print(first_5[key])
    terms = [term for term, count in tier_four_counts.items() if count > 0][::-1]
    counts = [count for term, count in tier_four_counts.items() if count > 0][::-1]
    sorted_counts = sorted(counts, reverse=True)
    # max_count = sorted_counts[1]
    max_count = sorted_counts[1]
    key_colors = {'physical_withdrawal_symptoms':'red', 'psychological_withdrawal_symptoms':'green'}
    term_colors = {}
    for category in category_term:
        for term in category_term[category]:
            if term in terms:
                if category in key_colors:
                    term_colors[term] = key_colors[category]
    plt.figure(figsize=(10, 6))
    plt.bar(terms, counts, color=[term_colors[term] for term in terms if term in term_colors])
    plt.xlabel('Term')
    plt.ylabel('Term Count')
    plt.xticks([i * 100 for i in range(len(terms) // 100 + 2)], [str(i * 100) for i in range(len(terms) // 100 + 2)])
    plt.yticks([i * 100 for i in range(max_count//100)], [str(i * 100) for i in range(max_count//100)])
    plt.ylim(0, max_count)
    legend_labels = {'physical_withdrawal_symptoms':'Physical Withdrawal Symptoms', 'psychological_withdrawal_symptoms':'Psychological Withdrawal Symptoms'}
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in key_colors.values()]
    plt.legend(legend_handles, [legend_labels[key] for key in key_colors.keys() if key], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_four_levels(term_count_file):
    category_term = build_category_term_dict('categories.csv')
    for i in range(1, 5):
        if i == 1:
            tier_one_terms = get_tier_level_terms('categories.csv', 1)
            tier_one_counts = count_tier_level_categories(term_count_file, tier_one_terms)
            plot_tier_one_category_counts(tier_one_counts, category_term)
        elif i == 2:
            tier_two_terms = get_tier_level_terms('categories.csv', 2)
            tier_two_counts = count_tier_level_categories(term_count_file, tier_two_terms)
            plot_tier_two_category_counts(tier_two_counts, category_term)
        elif i == 3:
            tier_three_terms = get_tier_level_terms('categories.csv', 3)
            tier_three_counts = count_tier_level_categories(term_count_file, tier_three_terms)
            plot_tier_three_category_counts(tier_three_counts, category_term)
        elif i == 4:
            tier_four_terms = get_tier_level_terms('categories.csv', 4)
            tier_four_counts = count_tier_level_categories(term_count_file, tier_four_terms)
            plot_tier_four_category_counts(tier_four_counts, category_term)

if __name__ == "__main__":
    pass
    client = MongoClient()
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

    # plot_four_levels('analysis_results/figure_48/figure_48_term_data.csv')
    client.close()