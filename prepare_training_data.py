import os
import csv
from sklearn.model_selection import train_test_split
import json

import parse

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def feature_encoding_to_binary(category, feature, encoded_label_list):
    if category == "tense":
        if feature == "present_tense":
            return 0 in encoded_label_list
        elif feature == "past_use":
            return 1 in encoded_label_list
        elif feature == "past_withdrawal":
            return 2 in encoded_label_list
        elif feature == "past_recovery":
            return 3 in encoded_label_list
        elif feature == "future_withdrawal":
            return 4 in encoded_label_list
    elif category == "atypical_information":
        if feature == "want_to_use":
            return 1 in encoded_label_list
        elif feature == "talking_about_withdrawal":
            return 2 in encoded_label_list
        elif feature == "talking_about_use":
            return 3 in encoded_label_list
        elif feature == "mentioning_withdrawal_drugs":
            return 4 in encoded_label_list
        elif feature == "not_mentioning_withdrawal":
            return 5 in encoded_label_list
    elif category == "special_cases":
        if feature == "relapse_mention":
            return 1 in encoded_label_list
        elif feature == "unintentional_withdrawal":
            return 2 in encoded_label_list
        elif feature == "abusing_subs":
            return 3 in encoded_label_list
        elif feature == "irregular_use":
            return 4 in encoded_label_list
        elif feature == "use_for_pain_relief":
            return 5 in encoded_label_list
    elif category == "use":
        if feature == "personal_regimen":
            return 1 in encoded_label_list
        elif feature == "improper_administration":
            return 2 in encoded_label_list
        elif feature == "purchase_of_drugs":
            return 4 in encoded_label_list
        elif feature == "negative_effects":
            return 5 in encoded_label_list
        elif feature == "activity_on_opiates":
            return 7 in encoded_label_list
        elif feature == "positive_effects":
            return 8 in encoded_label_list
    elif category == "withdrawal":
        if feature == "subs_method":
            return 1 in encoded_label_list
        elif feature == "methadone_method":
            return 2 in encoded_label_list
        elif feature == "zolpiclone_method":
            return 3 in encoded_label_list
        elif feature == "diazepam_method":
            return 4 in encoded_label_list
        elif feature == "kratom_method":
            return 5 in encoded_label_list
        elif feature == "unmentioned_method":
            return 6 in encoded_label_list
        elif feature == "xanax_method":
            return 7 in encoded_label_list
        elif feature == "sleeping_pills_method":
            return 8 in encoded_label_list
        elif feature == "loperamide_method":
            return 9 in encoded_label_list
        elif feature == "marijuana_method":
            return 12 in encoded_label_list
        elif feature == "gabapentin_method":
            return 13 in encoded_label_list
        elif feature == "klonopin_method":
            return 14 in encoded_label_list
        elif feature == "rhodiola_method":
            return 15 in encoded_label_list
        elif feature == "vivitrol_method":
            return 26 in encoded_label_list
        elif feature == "cigarette_methods":
            return 27 in encoded_label_list
        elif feature == "caffine_method":
            return 28 in encoded_label_list
        elif feature == "cold_turkey_method":
            return 29 in encoded_label_list
        elif feature == "ibogaine_method":
            return 20 in encoded_label_list
        elif feature == "restless_legs_symptom":
            return 10 in encoded_label_list
        elif feature == "sleep_disorder_symptom":
            return 11 in encoded_label_list
        elif feature == "GI_symptom":
            return 16 in encoded_label_list
        elif feature == "sweats_symptom":
            return 17 in encoded_label_list
        elif feature == "cold_sensitivity_symptom":
            return 18 in encoded_label_list
        elif feature == "nausea_vomiting_symptom":
            return 19 in encoded_label_list
        elif feature == "memory_loss_symptom":
            return 21 in encoded_label_list
        elif feature == "heartburn_symptom":
            return 22 in encoded_label_list
        elif feature == "headache_symptom":
            return 23 in encoded_label_list
        elif feature == "sore_throat_symptom":
            return 24 in encoded_label_list
        elif feature == "cold_flu_fever_symptom":
            return 25 in encoded_label_list
    elif category == "recovery":
        if feature == "offering_advice":
            return 1 in encoded_label_list
        elif feature == "challenges_through_recovery":
            return 2 in encoded_label_list
        elif feature == "danger_of_opiates":
            return 3 in encoded_label_list
    elif category == "co-use":
        if feature == "xanax":
            return 1 in encoded_label_list
        elif feature == "benzodiazepam":
            return 2 in encoded_label_list
        elif feature == "ambien":
            return 3 in encoded_label_list
        elif feature == "aderall":
            return 4 in encoded_label_list
        elif feature == "marijuana":
            return 5 in encoded_label_list
        elif feature == "cigarettes":
            return 6 in encoded_label_list
        elif feature == "cocaine":
            return 7 in encoded_label_list
        elif feature == "ketorolac":
            return 8 in encoded_label_list
        elif feature == "vinegar":
            return 9 in encoded_label_list
        elif feature == "alcohol":
            return 10 in encoded_label_list
        elif feature == "amphetamine":
            return 11 in encoded_label_list
        elif feature == "imodium":
            return 12 in encoded_label_list
    elif category == "off-topic":
        if feature == "public_health_awareness":
            return 1 in encoded_label_list
        elif feature == "seeking_community":
            return 5 in encoded_label_list
        elif feature == "other_persons_opiate_use":
            return 7 in encoded_label_list
        elif feature == "entertainment":
            return 8 in encoded_label_list
    elif category == "question":
        if feature == "opioid_use_lifestyle":
            return 1 in encoded_label_list
        elif feature == "technical_drug_use":
            return 2 in encoded_label_list
        elif feature == "effects":
            return 3 in encoded_label_list
        elif feature == "methadone":
            return 4 in encoded_label_list
        elif feature == "suboxone":
            return 5 in encoded_label_list
        elif feature == "improper_use":
            return 6 in encoded_label_list
        elif feature == "misc_question":
            return 7 in encoded_label_list
        elif feature == "subutex":
            return 32 in encoded_label_list
        elif feature == "tramadol":
            return 8 in encoded_label_list
        elif feature == "weed":
            return 25 in encoded_label_list
        elif feature == "kratom":
            return 26 in encoded_label_list
        elif feature == "darvocet":
            return 28 in encoded_label_list
        elif feature == "vivitrol":
            return 29 in encoded_label_list
        elif feature == "relate_to_defeated":
            return 11 in encoded_label_list
        elif feature == "relate_to_recovery":
            return 12 in encoded_label_list
        elif feature == "relate_to_withdrawal":
            return 20 in encoded_label_list
        elif feature == "relate_to_using":
            return 27 in encoded_label_list
        elif feature == "deal_with_relapse":
            return 29 in encoded_label_list
        elif feature == "recover_again":
            return 33 in encoded_label_list
        elif feature == "resetting_withrawal":
            return 18 in encoded_label_list
        elif feature == "withdrawal":
            return 13 in encoded_label_list
        elif feature == "withdrawal_symptoms":
            return 14 in encoded_label_list
        elif feature == "effects_of_withdrawal":
            return 15 in encoded_label_list
        elif feature == "withdrawal_pain":
            return 16 in encoded_label_list
        elif feature == "recovery_question":
            return 23 in encoded_label_list
        elif feature == "life_without_drugs":
            return 24 in encoded_label_list
        elif feature == "non-opiate_medication_question":
            return 17 in encoded_label_list
        elif feature == "NA_meeting_question":
            return 21 in encoded_label_list
        

def get_training_data(category, feature):
    encodings = parse.parse_feature_post_title_threshold(category)
    post_titles = []
    post_contents = []
    labels = []
    for post_id, post, title, state_label, feature_list in encodings:
        label = 1 if feature_encoding_to_binary(category, feature, feature_list) else 0
        post_titles.append(title)
        post_contents.append(post)
        labels.append(label)
    train_titles, val_titles, train_contents, val_contents, train_labels, val_labels = train_test_split(
    post_titles, post_contents, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_data = [
    {"post_title": title, "post_content": content, "label": label}
    for title, content, label in zip(train_titles, train_contents, train_labels)]
    val_data = [
    {"post_title": title, "post_content": content, "label": label}
    for title, content, label in zip(val_titles, val_contents, val_labels)]

    data_path = os.path.join("finetuning_data", category, feature)
    create_directory(data_path)
    with open(os.path.join(data_path, "train.jsonl"), "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
    with open(os.path.join(data_path, "validation.jsonl"), "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")

    
if __name__ == "__main__":
    get_training_data("withdrawal", "subs_method")