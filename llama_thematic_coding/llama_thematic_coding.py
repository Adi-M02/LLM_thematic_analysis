import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
import random
import time
import logging

url = "http://localhost:11434/api/chat"

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
    pass
        

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

def encode_tenses(output):
    process_tense(output, "present_tense", parse.parse_tense, thematically_encode_present_tense)
    process_tense(output, "past_use", parse.parse_tense, thematically_encode_past_use)
    process_tense(output, "past_withdrawal", parse.parse_tense, thematically_encode_past_withdrawal)
    process_tense(output, "past_recovery", parse.parse_tense, thematically_encode_past_recovery)
    process_tense(output, "future_withdrawal", parse.parse_tense, thematically_encode_future_withdrawal)

def encode_feature(output, encoding_type, thematic_encoding_function):
    if not os.path.exists(output):
        os.makedirs(output)
    csv_path = os.path.join(output, f"{encoding_type}_codes.csv")
    with open(csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", f"predicted_{encoding_type}", f"true_{encoding_type}"])
        writer.writeheader()
        encodings = parse.parse_feature(encoding_type)
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        valid_encodings = set()
        for encoding in encodings:
            post_id, post, title, state_label, feature = encoding
            if encoding_type == "incorrect_days_clean":
                feature = cast_incorrect_days_clean_to_binary(feature)
            valid_encodings.update(set(feature))
            try:
                thematic_code = thematic_encoding_function(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    f"predicted_{encoding_type}": thematic_code,
                    f"true_{encoding_type}": ','.join(map(str, feature))
                })
                file.flush()
                true_encodings.append(feature)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(
                        random.choice(
                            [
                                i for i in range(min(valid_encodings), max(valid_encodings)+1) 
                                if i not in feature]
                        )
                    )
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
    write_binary_classification_metrics(output, num_hallucinations, true_encodings, predicted_encodings)

if __name__ == "__main__":
  start = time.time()
  encode_tenses("llama_thematic_coding/12-1/tenses/run5")
  print(f"Time taken: {((time.time() - start)/60):.2f} minutes")