import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
import random
import time
import logging
from thematic_encoder import ThematicEncoder

url = "http://localhost:11434/api/chat"

category_feature_dict = {
  "tense": [
      "present_tense",
      "past_use",
      "past_withdrawal",
      "past_recovery",
      "future_withdrawal"
  ],
  "atypical_information": [
      "want_to_use",
      "talking_about_withdrawal",
      "talking_about_use",
      "mentioning_withdrawal_drugs",
      "not_mentioning_withdrawal"
  ],
  "special_cases": [
      "relapse_mention",
      "unintentional_withdrawal",
      "abusing_subs",
      "irregular_use",
      "use_for_pain_relief"
  ],
  "use": [
      "personal_regimen",
      "improper_administration",
      "purchase_of_drugs",
      "negative_effects",
      "activity_on_opiates",
      "positive_effects"
  ],
  "withdrawal": [
      "subs_method",
      "methadone_method",
      "zolpiclone_method",
      "diazepam_method",
      "kratom_method",
      "unmentioned_method",
      "xanax_method",
      "sleeping_pills_method",
      "loperamide_method",
      "marijuana_method",
      "gabapentin_method",
      "klonopin_method",
      "rhodiola_method",
      "vivitrol_method",
      "cigarette_methods",
      "caffine_method",
      "cold_turkey_method",
      "ibogaine_method",
      "restless_legs_symptom",
      "sleep_disorder_symptom",
      "GI_symptom",
      "sweats_symptom",
      "cold_sensitivity_symptom",
      "nausea_vomiting_symptom",
      "memory_loss_symptom",
      "heartburn_symptom",
      "headache_symptom",
      "sore_throat_symptom",
      "cold_flu_fever_symptom"
  ],
  "recovery": [
      "offering_advice",
      "challenges_through_recovery",
      "danger_of_opiates"
  ],
  "co-use": [
      "xanax",
      "benzodiazepam",
      "ambien",
      "aderall",
      "marijuana",
      "cigarettes",
      "cocaine",
      "ketorolac",
      "vinegar",
      "alcohol",
      "amphetamine",
      "imodium"
  ],
  "off-topic": [
      "public_health_awareness",
      "seeking_community",
      "other_persons_opiate_use", 
      "entertainment"
  ],
  "question": [
      "opioid_use_lifestyle",
      "technical_drug_use",
      "effects",
      "methadone",
      "suboxone",
      "improper_use",
      "subutex",
      "tramadol",
      "weed",
      "kratom",
      "darvocet",
      "vivitrol",
      "relate_to_defeated",
      "relate_to_recovery",
      "relate_to_withdrawal",
      "relate_to_using",
      "deal_with_relapse",
      "recover_again",
      "resetting_withrawal",
      "withdrawal",
      "withdrawal_symptoms",
      "effects_of_withdrawal",
      "withdrawal_pain",
      "recovery_question",
      "life_without_drugs",
      "non-opiate_medication_question",
      "NA_meeting_question"
  ]
}
feature_prompt_dict = {
        "present_tense": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if the language referring to the user's addiction state is in the present tense or has no tense.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if any language referring to the user's addiction state is in the past tense or future tense.
    - Provide a verbatim section of the text that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "past_use": """Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to the use of opioids is in the past tense, and the state label is 'withdrawal' or 'recovery'.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "past_withdrawal": """Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to the withdrawal from opioids is in the past tense, and the state label is 'use' or 'recovery'.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "past_recovery": """Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to recovery from opioids is in the past tense, and the state label is 'use' or 'withdrawal'.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "future_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to withdrawal from opioids is in the future tense.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "want_to_use": """""",
  "talking_about_withdrawal": """""",
  "talking_about_use": """""",
  "mentioning_withdrawal_drugs": """""",
  "not_mentioning_withdrawal": """""",
  "relapse_mention": """""",
  "unintentional_withdrawal": """""",
  "abusing_subs": """""",
  "irregular_use": """""",
  "use_for_pain_relief": """""",
  "personal_regimen": """""",
  "improper_administration": """""",
  "purchase_of_drugs": """""",
  "negative_effects": """""",
  "activity_on_opiates": """""",
  "positive_effects": """""",
  "subs_method": """""",
  "methadone_method": """""",
  "zolpiclone_method": """""",
  "diazepam_method": """""",
  "kratom_method": """""",
  "unmentioned_method": """""",
  "xanax_method": """""",
  "sleeping_pills_method": """""",
  "loperamide_method": """""",
  "marijuana_method": """""",
  "gabapentin_method": """""",
  "klonopin_method": """""",
  "rhodiola_method": """""",
  "vivitrol_method": """""",
  "cigarette_methods": """""",
  "caffine_method": """""",
  "cold_turkey_method": """""",
  "ibogaine_method": """""",
  "restless_legs_symptom": """""",
  "sleep_disorder_symptom": """""",
  "GI_symptom": """""",
  "sweats_symptom": """""",
  "cold_sensitivity_symptom": """""",
  "nausea_vomiting_symptom": """""",
  "memory_loss_symptom": """""",
  "heartburn_symptom": """""",
  "headache_symptom": """""",
  "sore_throat_symptom": """""",
  "cold_flu_fever_symptom": """""",
  "offering_advice": """""",
  "challenges_through_recovery": """""",
  "danger_of_opiates": """""",
  "xanax": """""",
  "benzodiazepam": """""",
  "ambien": """""",
  "aderall": """""",
  "marijuana": """""",
  "cigarettes": """""",
  "cocaine": """""",
  "ketorolac": """""",
  "vinegar": """""",
  "alcohol": """""",
  "amphetamine": """""",
  "imodium": """""",
  "public_health_awareness": """""",
  "seeking_community": """""",
  "other_persons_opiate_use": """""",
  "entertainment": """""",
  "opioid_use_lifestyle": """""",
  "technical_drug_use": """""",
  "effects": """""",
  "methadone": """""",
  "suboxone": """""",
  "improper_use": """""",
  "subutex": """""",
  "tramadol": """""",
  "weed": """""",
  "kratom": """""",
  "darvocet": """""",
  "relate_to_defeated": """""",
  "relate_to_recovery": """""",
  "relate_to_withdrawal": """""",
  "relate_to_using": """""",
  "deal_with_relapse": """""",
  "recover_again": """""",
  "resetting_withrawal": """""",
  "withdrawal": """""",
  "withdrawal_symptoms": """""",
  "effects_of_withdrawal": """""",
  "withdrawal_pain": """""",
  "recovery_question": """""",
  "life_without_drugs": """""",
  "non-opiate_medication_question": """""",
  "NA_meeting_question": """"""
}
  

def thematically_encode_present_tense(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
    "model": "llama3.2-vision:11b-instruct-q8_0",
    "format": "json",
    "options": {
        "temperature": 0.0
    },
    "messages": [
        {
            "role": "system",
            "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        },
        {
            "role": "user",
            "content": f"""
Instructions:

Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if the language referring to the user's addiction state is in the present tense or has no tense.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if any language referring to the user's addiction state is in the past tense or future tense.
    - Provide a verbatim section of the text that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
  State Label: {state_label}
  """
        }
    ],
    "stream": False
}
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_past_use(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
    "model": "llama3.2-vision:11b-instruct-q8_0",
    "format": "json",
    "options": {
        "temperature": 0.0
    },
    "messages": [
        {
            "role": "system",
            "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        },
        {
            "role": "user",
            "content": f"""
Instructions:

Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to the use of opioids is in the past tense, and the state label is 'withdrawal' or 'recovery'.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
  State Label: {state_label}
  """
        }
    ],
    "stream": False
}
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_past_withdrawal(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
    "model": "llama3.2-vision:11b-instruct-q8_0",
    "format": "json",
    "options": {
        "temperature": 0.0
    },
    "messages": [
        {
            "role": "system",
            "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        },
        {
            "role": "user",
            "content": f"""
Instructions:

Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to the withdrawal from opioids is in the past tense, and the state label is 'use' or 'recovery'.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
  State Label: {state_label}
  """
        }
    ],
      "stream": False
  }     
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_past_recovery(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
    "model": "llama3.2-vision:11b-instruct-q8_0",
    "format": "json",
    "options": {
        "temperature": 0.0
    },
    "messages": [
        {
            "role": "system",
            "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        },
        {
            "role": "user",
            "content": f"""
Instructions:

Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to recovery from opioids is in the past tense, and the state label is 'use' or 'withdrawal'.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
  State Label: {state_label}
  """
        }
    ],
    "stream": False
}
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_future_withdrawal(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
    "model": "llama3.2-vision:11b-instruct-q8_0",
    "format": "json",
    "options": {
        "temperature": 0.0
    },
    "messages": [
        {
          "role": "system",
          "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        },
        {
            "role": "user",
            "content": f"""
Instructions:

Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to withdrawal from opioids is in the future tense.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the above condition is not met.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
  - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
  State Label: {state_label}
  """
        }
    ],
    "stream": False
}
    response = requests.post(url, headers=headers, json=data)
    return response

def cast_incorrect_days_clean_to_binary(incorrect_days_clean):
    if incorrect_days_clean == [0]:
        return [0]
    elif incorrect_days_clean == [1]:
        return[1]
    elif incorrect_days_clean == [2]:
        return [1]

def write_binary_classification_metrics(output_dir, num_hallucinations, num_different_examples, true_encodings, predicted_encodings):
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "metrics_and_model.txt")
    with open(text_path, 'w') as f:
        f.write(f"hallucinations/errors: {num_hallucinations}\n")
        f.write(f"responses with some hallucinated portion: {num_different_examples}\n")
        f.write(f"Encodings:\n")
        f.write(f"- True Encodings:\n")
        f.write(f"    - Class 0: {true_encodings.count(0)}\n")
        f.write(f"    - Class 1: {true_encodings.count(1)}\n")
        f.write(f"- Predicted Encodings:\n")
        f.write(f"    - Class 0: {predicted_encodings.count(0)}\n")
        f.write(f"    - Class 1: {predicted_encodings.count(1)}\n\n")
        try:
            f.write(f"Performance Metrics:\n")
            f.write(f"- Accuracy: {accuracy_score(true_encodings, predicted_encodings):.4f}\n")
            f.write(f"- Macro Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"- Weighted Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(true_encodings, predicted_encodings)
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            f.write(f"    [[TP: {tp} ({(tp / total) * 100:.2f}%), FP: {fp} ({(fp / total) * 100:.2f}%)]\n")
            f.write(f"     [FN: {fn} ({(fn / total) * 100:.2f}%), TN: {tn} ({(tn / total) * 100:.2f}%)]]\n")
        except Exception as e:
            f.write(f"Error calculating metrics: {e}\n")
    
    print(f"Metrics written to {text_path}")

def tense_type_condition(tense_list, tense_type):
    if tense_type == "present_tense":
        if 0 in tense_list:
            return True
    elif tense_type == "past_use":
        if 1 in tense_list:
            return True
    elif tense_type == "past_withdrawal":
        if 2 in tense_list:
            return True
    elif tense_type == "past_recovery":
        if 3 in tense_list:
            return True
    elif tense_type == "future_withdrawal":
        if 4 in tense_list:
            return True
    return False

def tense_log_identifier(log_file_path):
    return os.path.basename(log_file_path).replace(".txt", "")

def setup_logging(log_file_path):
    # Configure logging for each tense with a unique log file
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as file:
            pass  # Create an empty file if it does not exist
    logger = logging.getLogger(tense_log_identifier(log_file_path))
    logger.setLevel(logging.INFO)

    # Remove previous handlers if they exist to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set up file handler with the given log file path
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)

    # Set log format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

def create_directory(directory_path):
    # Create a directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def tense_type_condition(tense_list, tense_type):
    # Return True or False based on the given tense type condition
    if tense_type == "present_tense":
        if 0 in tense_list:
            return True
    elif tense_type == "past_use":
        if 1 in tense_list:
            return True
    elif tense_type == "past_withdrawal":
        if 2 in tense_list:
            return True
    elif tense_type == "past_recovery":
        if 3 in tense_list:
            return True
    elif tense_type == "future_withdrawal":
        if 4 in tense_list:
            return True
    return False

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

def compare_example_and_post(llm_output):
    modified_llm_output = []
    num_diff = 0
    with open(llm_output, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            post = parse.get_post_title_string(row['post_id'])
            if not post:
                continue
            if row['verbatim_example']:
              if row['verbatim_example'].lower() in post.lower():
                  row["exact_match"] = "True"
              elif row['verbatim_example'] == "ERROR":
                  row["exact_match"] = "ERROR"
              elif row['verbatim_example'].strip() == "None":
                    row["exact_match"] = "True"
              else:
                  row["exact_match"] = "False"
                  num_diff += 1
            else:
              row["exact_match"] = "ERROR"
            modified_llm_output.append(row)
    with open(llm_output, 'w', newline='', encoding="utf-8") as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      for row in modified_llm_output:
          filtered_row = {key: row.get(key, '') for key in fieldnames}
          writer.writerow(filtered_row)
    return num_diff

def encode_tenses(output):
    process_tense(output, "present_tense", parse.parse_tense, thematically_encode_present_tense)
    process_tense(output, "past_use", parse.parse_tense, thematically_encode_past_use)
    process_tense(output, "past_withdrawal", parse.parse_tense, thematically_encode_past_withdrawal)
    process_tense(output, "past_recovery", parse.parse_tense, thematically_encode_past_recovery)
    process_tense(output, "future_withdrawal", parse.parse_tense, thematically_encode_future_withdrawal)

def process_tense(output, tense_type, parse_function, encode_function):
    # Create folder for the tense type
    directory_path = os.path.join(output, tense_type)
    create_directory(directory_path)
    # Set up a unique log for each tense type
    log_file_path = os.path.join(directory_path, f"{tense_type}_error_log.txt")
    logger = setup_logging(log_file_path)
    # Log processing information
    logger.info(f"{tense_type.upper()}: \n")
    # Set CSV path and create CSV file with headers
    csv_path = os.path.join(directory_path, f"{tense_type}_codes.csv")
    with open(csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense", "verbatim_example", "exact_match"])
        writer.writeheader()
        # Parsing encodings
        encodings = parse_function()
        true_encodings = []
        predicted_encodings = []
        num_errors = 0
        # Iterate over encodings and process
        for encoding in encodings:
            file.flush()
            post_id, post, title, state_label, tense_list = encoding
            try:
                if tense_type_condition(tense_list, tense_type):
                    true_tense = 1
                else:
                    true_tense = 0
                response = encode_function(state_label, post, title)
                try:
                    thematic_code_json = json.loads(response.json()['message']['content'])
                    thematic_code = thematic_code_json['label']
                    try:
                      verbatim_example = thematic_code_json['language']
                    except:
                      verbatim_example = "None"
                    writer.writerow({
                        "post_id": post_id,
                        "predicted_tense": thematic_code,
                        "verbatim_example": verbatim_example,
                        "true_tense": true_tense
                    })
                except Exception as e:
                    num_errors += 1
                    writer.writerow({
                        "post_id": post_id,
                        "predicted_tense": "ERROR",
                        "verbatim_example": "ERROR",
                        "true_tense": true_tense
                    })
                    logger.error(f"JSON error: {e}, post id: {post_id}, response: {response.json()}")
                    continue
                try:
                    predicted_encodings.append(int(thematic_code))
                    true_encodings.append(true_tense)
                except Exception as e:
                    num_errors += 1
                    logger.error(f"Error appending: post id: {post_id}, {thematic_code}")
            except Exception as e:
                num_errors += 1
                logger.error(f"Error processing encoding {encoding}: {e}")
        num_different_examples = compare_example_and_post(csv_path)
        write_binary_classification_metrics(directory_path, num_errors, num_different_examples, true_encodings, predicted_encodings)

def verbatim_example_matches(logger, post_id, verbatim_example):
    post = parse.get_post_title_string(logger, post_id)
    if not post:
        return False
    if verbatim_example.lower() in post.lower():
        return True
    elif verbatim_example.strip() == "None":
        return True
    elif verbatim_example == "ERROR":
        return True
    return False
    
    

def write_response_update_evaluation_lists(writer, logger, response, post_id, true_tense, num_errors, predicted_encodings, true_encodings):
    try:
        thematic_code_json = json.loads(response.json()['message']['content'])
        thematic_code = thematic_code_json['label']
        try:
            verbatim_example = thematic_code_json['language']
        except:
            verbatim_example = "None"
        writer.writerow({
            "post_id": post_id,
            "predicted_tense": thematic_code,
            "verbatim_example": verbatim_example,
            "true_tense": true_tense, 
        })
    except Exception as e:
        num_errors += 1
        writer.writerow({
            "post_id": post_id,
            "predicted_tense": "ERROR",
            "verbatim_example": "ERROR",
            "true_tense": true_tense
        })
        logger.error(f"JSON error: {e}, post id: {post_id}, response: {response.json()}")
    try: 
        predicted_encodings.append(int(thematic_code))
        true_encodings.append(true_tense)


def encode_features(output, category_feature_dict = category_feature_dict):
    for category in category_feature_dict:
        directory_path = os.path.join(output, category)
        encoder = ThematicEncoder()
        create_directory(directory_path)
        log_file_path = os.path.join(directory_path, "error_log.txt")
        logger = setup_logging(log_file_path)
        for feature in category_feature_dict[category]:
            feature_directory = os.path.join(directory_path, feature)
            create_directory(feature_directory)
            csv_path = os.path.join(feature_directory, f"{feature}_codes.csv")
            with open(csv_path, 'w', newline='', encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense", "verbatim_example", "exact_match"])
                writer.writeheader()
                encodings = parse.parse_feature(category)
                for encoding in encodings:
                    post_id, post, title, state_label, tense_list = encoding
                    true_tense = 1 if feature_encoding_to_binary(category, feature, tense_list) else 0
                    response = encoder.encode(feature_prompt_dict[feature], post, title, state_label)
                    




            
if __name__ == "__main__":

      
  print(f"Time taken: {((time.time() - start)/60):.2f} minutes")