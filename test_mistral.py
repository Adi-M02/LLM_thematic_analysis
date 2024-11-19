import json
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pymongo import MongoClient
import test_database as tdb
import mongo_database as mongo_db
import torch


def RBIC_mistral(model, tokenizer, device, title, post):
    #first prompt
    messages = [
        SystemMessage(content="Your role is to understand the cause and effect relationships in social media posts."),
        UserMessage(content="Can you provide a brief definition of what a cause-effect relationship is?")
    ]
    completion_request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=256, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    first_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    # Add the model's response to the message history
    messages.append(AssistantMessage(content=first_result))

    #second prompt
    messages.append(UserMessage(content="Based on your role, can you explain the term 'causal gist' in relation to sentences that have causal coherence?"))
    completion_request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=256, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    second_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    # Add the model's response to the message history
    messages.append(AssistantMessage(content=second_result))

    #third prompt
    messages.append(UserMessage(content=f"So, based on the title: '{title}' and post: '{post}', is there a clear and well-defined cause-effect relationship? If the relationship is clear, answer: 'Yes'. If you are unsure, just answer: 'No'. Do not give any explanations."))
    completion_request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=8, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    third_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    if third_result.lower() not in ["yes", "yeah", "yep", "yup", "sure", "indeed", "correct", "right", "affirmative", "positive", "true"]:
        return False
    # Add the model's response to the message history
    messages.append(AssistantMessage(content=third_result))

    #fourth prompt
    messages.append(UserMessage(content='Indeed, there is a cause-effect relationship in the given post. Extract the corresponding cause phrase and effect phrase in the given post. Just respond in JSON format with a single JSON: {"Cause": "", "Effect": ""}'))
    completion_request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=256, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    fourth_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    # Add the model's response to the message history
    messages.append(AssistantMessage(content=fourth_result))
    #get json
    try:
        fourth_result_json = json.loads(fourth_result)
        generated_cause = fourth_result_json['Cause']
        generated_effect = fourth_result_json['Effect']
    except Exception as e:
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Title: {title}\n")
            log_file.write(f"Post: {post}\n")
            log_file.write(f"Fourth Result: {fourth_result}\n")
            log_file.write(f"Error: {e}\n")
        return False

    #fifth promptitle = html.unescape(title)
    messages.append(UserMessage(content=f"Generate a reasonable and clear one sentence causal gist without specifics based on 'Cause': '{generated_cause}' and 'Effect': '{generated_effect}' and your understanding of the post with the cause-effect relationship. Only provide the sentence."))
    completion_request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=128, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    fifth_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    # Add the model's fourth response to the message history
    messages.append(AssistantMessage(content=fifth_result))
    return str(fifth_result)

def test_RBIC_mistral(db, collection):
    posts = tdb.get_posts_in_subreddits(collection, ["opiatesrecovery", "opiates"])
    new_collection = db['gist_test']
    i = 0
    mistral_models_path = '/local/disk2/not_backed_up/mistral_models/7B-Instruct-v0.3/'
    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer.from_folder(mistral_models_path)
    model.to(device)
    for post in posts:
        i+=1
        if i%10 == 0:
            print(f"Processed {i} posts")
        if i%100 == 0:
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"Processed {i} posts\n")
        title = post["title"]
        selftext = post["selftext"]
        selftext = mongo_db.preprocess_post(selftext)
        if not selftext or selftext == "[removed]":
            continue
        mistral_output = RBIC_mistral(model, tokenizer, device, title, selftext)
        # if mistral_output:
        #     print(f"Title: {title}\nPost: {selftext}\nMistral Output: {mistral_output}\n")
        # else:
        #     print(f"Title: {title}\nPost: {selftext}\nMistral Output: None\n")
        # continue
        new_obj = post.copy()
        if mistral_output:
            new_obj["has_gist"] = True
            new_obj["gist"] = mistral_output
        else:
            new_obj["has_gist"] = False
            new_obj["gist"] = ""
        new_obj.pop('_id', None)
        try:
            new_collection.insert_one(new_obj)
        except Exception as e:
            print("Failed to insert post")
            print(e)


if __name__ == "__main__":
    client = MongoClient()
    db = client['reddit']
    collection = db['posts_and_comments']
    test_RBIC_mistral(db, collection)