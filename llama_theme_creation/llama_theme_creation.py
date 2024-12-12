import json
import time
import logging
import csv
import os
import random
import parse_codings_themes as parse
from theme_creator import ThemeCreator
from theme_creator_feed_forward import ThemeCreatorFeedForward
from theme_creator_feed_forward_with_desc import ThemeCreatorFeedForwardDesc
from theme_creator_generalizer import ThemeCreatorGeneralizer
from theme_creator_no_feedforward import ThemeCreatorNoFeedForward

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
    completed_posts = set()  # Tructure(self, writer):
    #   prompt = f"""
    #     data = {{
    #         {self.default_data.copy()},
    #     "messages": [
    #         {{"role:"system", "content"{self.system_message}}},
    #         {{"role": "user", "content": 
    #           Instructions:

    #           Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only by appending new major themes related to opiate addiction state classification to the input list. Only append if the new theme is not similar to an existing theme. If a new theme is similar to an existing theme combine the themes into a new more general theme. If no new themes are identified return the input list of themes unmodified. Do not include any additional descriptions, reasoning, or text in your response.

    #           - Important Notes:
    #             - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
    #           - Definitions of Addiction States:
    #             - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    #             - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    #             - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

    #           - Response Format:
    #           {{
    #             "themes": [ructure(self, writer):
    #   prompt = f"""
    #     data = {{
    #         {self.default_data.copy()},
    #     "messages": [
    #         {{"role:"system", "content"{self.system_message}}},
    #         {{"role": "user", "content": 
    #           Instructions:

    #           Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only by appending new major themes related to opiate addiction state classification to the input list. Only append if the new theme is not similar to an existing theme. If a new theme is similar to an existing theme combine the themes into a new more general theme. If no new themes are identified return the input list of themes unmodified. Do not include any additional descriptions, reasoning, or text in your response.

    #           - Important Notes:
    #             - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
    #           - Definitions oructure(self, writer):
    #   prompt = f"""
    #     data = {{
    #         {self.default_data.copy()},
    #     "messages": [
    #         {{"role:"system", "content"{self.system_message}}},
    #         {{"role": "user", "content": 
    #           Instructions:

    #           Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only by appending new major themes related to opiate addiction state classification to the input list. Only append if the new theme is not similar to an existing theme. If a new theme is similar to an existing theme combine the themes into a new more general theme. If no new themes are identified return the input list of themes unmodified. Do not include any additional descriptions, reasoning, or text in your response.

    #           - Important Notes:
    #             - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
    #           - Definitions of Addiction States:
    #             - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    #             - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    #             - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

    #           - Response Format:
    #           {{
    #             "themes": [
    #               "Title of theme","Title of another theme", ...
    #             ]
    #           }}
                
    #           - Respond based on the following inputs:
    #             Post: {{post}}
    #             Post Title: {{title}}
    #           }}
    #         }}
    # """f Addiction States:
    #             - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    #             - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    #             - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

    #           - Response Format:
    #           {{
    #             "themes": [
    #               "Title of theme","Title of another theme", ...
    #             ]
    #           }}
                
    #           - Respond based on the following inputs:
    #             Post: {{post}}
    #             Post Title: {{title}}
    #           }}
    #         }}
    # """
    #               "Title of theme","Title of another theme", ...
    #             ]
    #           }}
                
    #           - Respond based on the following inputs:
    #             Post: {{post}}
    #             Post Title: {{title}}
    #           }}
    #         }}
    # """rack completed post IDs
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
                    response1 = theme_creator1.create_themes(post, title)
                    response2 = theme_creator2.create_themes(post, title)
                    response3 = theme_creator3.create_themes(post, title)
                    response4 = theme_creator4.create_themes(post, title)
                    responses = [response1, response2, response3, response4]
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
    themes = set()
    # create all themes
    i = 0
    for post_id, post, title in posts_and_titles:
        i += 1
        print(i)
        try:
            print(f"num themes: {len(themes)}")
            response = creator.create_themes(post, title, list(themes))
            themes_json = json.loads(response.json()['message']['content'])
            themes.update(themes_json['themes'])
            print(len(themes))
        except:
            continue
    cur_length = len(themes)
    output_file = os.path.join(output, "feedforward_themes.txt")
    with open(output_file, "w") as file:
        for theme in themes:
            file.write(f"{theme}\n")

    # while 
def theme_creation_feedforward_desc(output):
    create_directory(output)
    posts_and_titles = parse.get_posts_and_titles_only()
    creator = ThemeCreatorFeedForwardDesc()
    themes = []
    i = 0
    for post_id, post, title in posts_and_titles:
        i += 1
        print(i)
        try:
            print(f"num themes: {len(themes)}")
            response = creator.create_themes(post, title, list(themes))
            themes_json = json.loads(response.json()['message']['content'])
            themes.append(themes_json['themes'])
            
        except:
            continue

    output_file = os.path.join(output, "feedforward_desc_themes.txt")
    with open(output_file, "w") as file:
        for theme in themes:
            file.write(f"{theme}\n")

def generalize_themes(themes_file, output):
    create_directory(output)
    with open(themes_file, "r") as file:
        themes = [line.strip() for line in file.readlines()]
    generalizer = ThemeCreatorGeneralizer()
    posts_and_titles = parse.get_posts_and_titles_only()
    major_themes = []
    out_file = os.path.join(output, "generalized_themes.txt")
    with open(out_file, "w") as file:
        i = 0
        for post_id, post, title in posts_and_titles:
            i += 1
            print(i)
            file.flush()
            try:
                response = generalizer.generalize_themes(post, title, themes, major_themes)
                themes_json = json.loads(response.json()['message']['content'])
                file.write(f"Post ID: {post_id}\n")
                for theme in themes_json['themes']:
                    file.write(f"    Theme: {theme['theme']}")
                    file.write(f"    Description: {theme['description']}\n")
            except Exception as e:
                file.write(f"Post ID: {post_id} error {e}\n")

def theme_creation_no_ff(url, output):
    create_directory(output)
    posts_and_titles = parse.get_posts_and_titles_only()
    creator = ThemeCreatorNoFeedForward(url)
    themes = []
    i = 0
    for post_id, post, title in posts_and_titles:
        i += 1
        print(i)
        try:
            response = creator.create_themes(post, title, list(themes))
            themes_json = json.loads(response.json()['message']['content'])
            themes.append(themes_json['themes'])
            
        except:
            continue

    output_file = os.path.join(output, "feedforward_desc_themes.txt")
    with open(output_file, "w") as file:
        for theme in themes:
            file.write(f"{theme}\n")
    
if __name__ == "__main__":
    start = time.time()
    generalize_themes('llama_theme_creation/12-10/feedforward_themes.txt', 'llama_theme_creation/12-11/generalized_themes/run3')
    # theme_creation_feedforward_desc("llama_theme_creation/12-10")
    # theme_creation_feedforward_themes("llama_theme_creation/12-10")
    print(f"Time taken: {((time.time() - start) / 60):.2f} minutes")