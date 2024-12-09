import json
import time
import logging
import csv
import os
import random
import parse_codings_themes as parse
from theme_creator import ThemeCreator
from theme_creator_feed_forward import ThemeCreatorFeedForward

def tense_log_identifier(log_file_path):
    return os.path.basename(log_file_path).replace(".txt", "")

def setup_logging(log_file_path):
    # Get a unique logger name based on the log file path
    logger_name = tense_log_identifier(log_file_path)
    logger = logging.getLogger(logger_name)
    # If the logger is already set up, return it
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.INFO)
    # Set up file handler with the given log file path
    handler = logging.FileHandler(log_file_path, mode='a')  # Use 'a' to append logs
    handler.setLevel(logging.INFO)
    # Set log format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(handler)
    return logger

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def create_themes(output, sample_size=5):
    create_directory(output)
    posts_and_titles = parse.get_posts_and_titles_only()
    total_required = 50
    sampled_elements = random.sample(posts_and_titles, total_required)  # Initial random sample
    completed_posts = set()  # Track completed post IDs
    chunks = [sampled_elements[i:i + sample_size] for i in range(0, len(sampled_elements), sample_size)]
    output_file = os.path.join(output, "themes.txt")
    theme_creator1 = ThemeCreator("llama3.3:70b")
    theme_creator2 = ThemeCreator("llama3.3:70b")
    theme_creator3 = ThemeCreator("llama3.2-vision:11b-instruct-q8_0")
    theme_creator4 = ThemeCreator("llama3.2-vision:11b-instruct-q8_0")
    
    retry_pool = list(set(posts_and_titles) - set(sampled_elements))  # Posts available for retries

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output, f"chunk_{i}_themes.txt")
        with open(output_file, "w") as file:
            for post_id, post, title in chunk:
                if post_id in completed_posts:
                    continue  # Skip posts that are already completed

                try:
                    # Attempt to generate themes
                    # response1 = theme_creator1.create_themes(post, title)
                    # response2 = theme_creator2.create_themes(post, title)
                    response3 = theme_creator3.create_themes(post, title)
                    response4 = theme_creator4.create_themes(post, title)
                    responses = [response3, response4]

                    # Write themes
                    write_theme_and_human_themes(file, post_id, responses)
                    completed_posts.add(post_id)  # Mark as completed

                except Exception as e:
                    print(f"Error processing post {post_id}: {e}")
                    if retry_pool:
                        # Replace with a new post from the retry pool
                        new_post = retry_pool.pop()
                        chunk.append(new_post)  # Add it to the current chunk
                        print(f"Retrying with new post: {new_post[0]}")

    write_model(output, theme_creator3)


def write_model(output_dir, creator):
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "model.txt")
    with open(text_path, "w") as file:
        creator.write_prompt_structure(file)


def write_theme_and_human_themes(file, post_id, responses):
    file.write(f"Post ID: {post_id}\n")
    file.write("Human generated themes:\n")
    for theme, value in parse.get_post_theme_presence(post_id).items():
        file.write(f"Theme: {theme}  value: {value} ")
    file.write("LLM generated themes:\n")
    for i, response in enumerate(responses):
        try:
            themes_json = json.loads(response.json()['message']['content'])
        except:
            continue
        file.write(f"theme creator {i+1} \n")
        for theme in themes_json['themes']:
            file.write(f"Theme: {theme['theme']}  Description: {theme['description']} Example: {theme['example']}\n")
        file.write("\n\n")

def theme_creation_feedforward_themes(output):
    create_directory(output)
    posts_and_titles = parse.get_posts_and_titles_only()
    creator = ThemeCreatorFeedForward()
    themes = []
    for post_id, post, title in posts_and_titles:
        try:
            response = creator.create_themes(post, title)
        themes.append((post_id, response))
    
if __name__ == "__main__":
    start = time.time()
    create_themes("llama_theme_creation/12-8/run1")
    print(f"Time taken: {((time.time() - start) / 60):.2f} minutes")