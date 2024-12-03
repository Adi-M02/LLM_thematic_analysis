import requests

class ThematicEncoder:
    def __init__(self):
        self.url = "http://localhost:11434/api/chat"
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": "llama3.2-vision:11b-instruct-q8_0",
            "format": "json",
            "options": {"temperature": 0.0},
            "stream": False,
        }
        self.system_message = (
            "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        )

    def encode(self, instructions, state_label, post, title):
        user_message = f"""
Instructions:

{instructions}

- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
  State Label: {state_label}
"""
        
        data = self.default_data.copy()
        data["messages"] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        response = requests.post(self.url, headers=self.headers, json=data)
        return response

# Example usage:
encoder = ThematicEncoder()

instructions_present_tense = ""
response = encoder.encode(instructions_present_tense, "use", "I am currently using opioids.", "Ongoing struggle with addiction.")
