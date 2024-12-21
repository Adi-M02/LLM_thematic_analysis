import requests

class Encoder:
    def __init__(self, model):
        self.url = "http://localhost:11434/api/chat"
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": model,
            "stream": False,
        }
        # self.system_message = (
        #     "You are an academic researcher analyzing opiate use related social media posts. Your task is to examine the language in posts and post titles, identify semantic themes, and classify the language based on specific rules related to context and thematic content. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        # )

    def encode(self, post, title):
        user_message = f"""
{title} {post}
"""
        
        data = self.default_data.copy()
        data["messages"] = [
            # {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message}
        ]

        response = requests.post(self.url, headers=self.headers, json=data)
        return response