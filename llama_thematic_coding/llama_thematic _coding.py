import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv

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
                    "content": "Your role is to take part in an academic research study by understanding posts related to opiate use on Reddit and thematically coding posts based on the content of the post. Strictly follow these instructions: respond with exactly '0', '1', or '2'. Do not describe your reasoning. Do not provide explanations. Do not include any extra text."
                },
                {
                    "role": "user",
                    "content": f"""Based on the following inputs:
                        - Post: {post}
                        - Title: {title}
                        - State label: {state_label}

                        Please thematically encode the number of days clean in the post according to these rules:
                        1. If the state label is 'use', you must always respond with '0'. This rule overrides all others.
                        2. If an accurate number of days clean is mentioned, respond with '0'.
                        3. If the number of days clean is mentioned but not accurate, respond with '1'.
                        4. If the number of days clean is not mentioned, unclear, or multiple numbers of days clean are mentioned, respond with '2'.

                        Respond with a single digit: '0', '1', or '2'. Do not include any other text or reasoning."""
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
                    "content": "Your role is to take part in an academic research study by understanding posts related to opiate use on Reddit and thematically coding posts based on the content of the post. Strictly follow these instructions: respond with exactly '0', '1', or '2'. Do not describe your reasoning. Do not provide explanations. Do not include any extra text."
                },
                {
                    "role": "user",
                    "content": f"""Based on the following inputs:
                        - Title: {title}
                        - State label: {state_label}

                        Please thematically encode the number of days clean in the post according to these rules:
                        1. If the state label is 'use', you must always respond with '0'. This rule overrides all others.
                        2. If an accurate number of days clean is mentioned, respond with '0'.
                        3. If the number of days clean is mentioned but not accurate, respond with '1'.
                        4. If the number of days clean is not mentioned, unclear, or multiple numbers of days clean are mentioned, respond with '2'.

                        Respond with a single digit: '0', '1', or '2'. Do not include any other text or reasoning."""
                }
            ],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

if __name__ == "__main__":
    out = parse.parse_csv()
    # with open('llama_thematic_coding/incorrect_days_clean_codes.csv', 'w') as f:
    #     f.write("post_id,incorrect_days_clean\n")

    # for post_id, post, title, state_label in out:
    #     thematic_code = thematically_encode_days_clean(state_label, title, post)
    #     with open('llama_thematic_coding/incorrect_days_clean_codes.csv', 'a') as f:
    #         f.write(f"{post_id},{thematic_code}\n")
    #         pass
    llm_codings = []
    with open('llama_thematic_coding/incorrect_days_clean_codes.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',', skipinitialspace=True)
        
        for row in reader:
            value = row['incorrect_days_clean_LLM']
            try:
                # Try casting to int
                llm_codings.append(int(value))
            except:
                llm_codings.append(1)
                # print(row['post_id'])
                continue
    
    true_codings = []
    with open('/Users/adimukundan/Downloads/Thematic Analysis Opiate Subreddits/All_Codes_Manual_Analysis_fixEncoding.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            value = row['incorrect days clean']
            try:
                # Try casting to int
                true_codings.append(int(value))
            except:
                true_codings.append(-1)
                print("true encoding", row['Post ID'])
                continue
    
    print(true_codings.count(0), true_codings.count(1), true_codings.count(2))
    print(llm_codings.count(0), llm_codings.count(1), llm_codings.count(2))
    
    print(f"f1 macro: {f1_score(true_codings, llm_codings, average='macro')}")
    print(f"accuracy: {accuracy_score(true_codings, llm_codings)}")
    print(f"precision: {precision_score(true_codings, llm_codings, average='macro')}")
    print(f"recall: {recall_score(true_codings, llm_codings, average='macro')}")
    
    
