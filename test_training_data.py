import json

def process_jsonl_file(file_path):
    max_words = 0
    max_title = ""
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            post = data.get('post_content', '')
            title = data.get('post_title', '')
            word_count = len(post) + len(title)
            
            if word_count > max_words:
                max_words = word_count
                max_title = title
    
    return max_title, max_words

def main():
    jsonl_files = ['finetuning_data/withdrawal/subs_method/train.jsonl', 'finetuning_data/withdrawal/subs_method/validation.jsonl']  # Replace with your actual file names
    overall_max_title = ""
    overall_max_words = 0
    
    for file_path in jsonl_files:
        title, words = process_jsonl_file(file_path)
        if words > overall_max_words:
            overall_max_words = words
            overall_max_title = title
    
    print(f"Post title with the most words: {overall_max_title} ({overall_max_words} words)")

if __name__ == "__main__":
    main()