import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os

url = "http://localhost:11434/api/chat"

def thematically_encode_days_clean(state_label, title, post=None):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are participating in an academic research study focused on posts about opiate use on Reddit. Your task is to classify posts into two categories based on whether the number of days clean is explicitly mentioned. Strictly follow the rules provided and respond with only '0' or '1'. Do not include reasoning, explanations, or additional text in your responses."
    },
    {
        "role": "user",
        "content": f"""Based on the following inputs:
Post: {post}
Title: {title}
State label: {state_label}

Follow these rules to classify the post:

1. If the state label is 'use', respond with '0'.

2. If the post includes a single, clear number of current days clean (e.g., 'I have been clean for 30 days'), respond with '0'.

3. If the post does not mention days clean, mentions multiple numbers of days clean, includes vague terms (e.g., 'a few days', 'some time'), or the number of days clean is unclear, respond with '1'.

Note: Ignore mentions of future intentions (e.g., 'I will be clean for 30 days') or past periods of being clean (e.g., 'I was clean for 30 days last year') unless they state the current number of days clean.

Respond with exactly one digit: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are participating in an academic research study focused on posts about opiate use on Reddit. Your task is to classify post titles into two categories based on whether the number of days clean is explicitly mentioned. Strictly follow the rules provided and respond with only '0' or '1'. Do not include reasoning, explanations, or additional text in your responses."
    },
    {
        "role": "user",
        "content": f"""Based on the following inputs:
Post title: {title}
State label: {state_label}

Note: Only the post title is available; there is no additional post content.

Follow these rules to classify the post:

1. **Priority Rule**: If the state label is 'use', respond with '0'.

2. If the post title includes a single, clear number of current days clean (e.g., 'I have been clean for 30 days'), respond with '0'.

3. If the post title does not mention days clean, mentions multiple numbers of days clean, includes vague terms (e.g., 'a few days', 'some time'), uses metaphorical language, or the number of days clean is unclear, respond with '1'.

Note: Ignore mentions of future intentions (e.g., 'I will be clean for 30 days') or past periods of being clean (e.g., 'I was clean for 30 days last year') unless they state the current number of days clean.

Respond with exactly one digit: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

def cast_incorrect_days_clean_to_binary(incorrect_days_clean):
    if incorrect_days_clean == '0':
        return 0
    elif incorrect_days_clean == '1':
        return 1
    elif incorrect_days_clean == '2':
        return 1

def encode_incorrect_days(output):
    if not os.path.exists(output):
        os.makedirs(output)
    csv_path = os.path.join(output, "incorrect_days_clean_codes.csv")
    with open(csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_incorrect_days_clean", "true_incorrect_days_clean"])
        writer.writeheader()
        encodings = parse.parse_csv()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            try:
                post_id, post, title, state_label, incorrect_days_clean = encoding
                thematic_code = thematically_encode_days_clean(state_label, title, post)
                binary_true_incorrect_days_clean = cast_incorrect_days_clean_to_binary(incorrect_days_clean)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_incorrect_days_clean": thematic_code,
                    "true_incorrect_days_clean": binary_true_incorrect_days_clean
                })
                file.flush()
                true_encodings.append(binary_true_incorrect_days_clean)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding isn't a number, add a wrong entry to the predictions
                    if binary_true_incorrect_days_clean == 0:
                        predicted_encodings.append(1)
                    else:
                        predicted_encodings.append(0)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
    text_path = os.path.join(output, "metrics_and_model.txt")
    with open(text_path, 'w') as f:
        f.write(f"hallucinations: {num_hallucinations}\n")
        f.write(f"true encodings: 0: {true_encodings.count(0)}, 1: {true_encodings.count(1)}\n")
        f.write(f"predicted encodings: 0: {predicted_encodings.count(0)}, 1: {predicted_encodings.count(1)}\n")
        f.write(f"accuracy: {accuracy_score(true_encodings, predicted_encodings)}\n")
        f.write(f"f1 macro: {f1_score(true_encodings, predicted_encodings, average='macro')}\n")
        f.write(f"precision macro: {precision_score(true_encodings, predicted_encodings, average='macro')}\n")
        f.write(f"recall macro: {recall_score(true_encodings, predicted_encodings, average='macro')}\n")
        f.write(f"f1 weighted: {f1_score(true_encodings, predicted_encodings, average='weighted')}\n")
        f.write(f"precision weighted: {precision_score(true_encodings, predicted_encodings, average='weighted')}\n")
        f.write(f"recall weighted: {recall_score(true_encodings, predicted_encodings, average='weighted')}\n")


if __name__ == "__main__":
    encode_incorrect_days('llama_thematic_coding/11-20_incorrect_days')
   
    
    # print(f"f1 macro: {f1_score(true_codings, llm_codings, average='macro')}")
    # print(f"accuracy: {accuracy_score(true_codings, llm_codings)}")
    # print(f"precision: {precision_score(true_codings, llm_codings, average='macro')}")
    # print(f"recall: {recall_score(true_codings, llm_codings, average='macro')}")
    
    
