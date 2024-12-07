import requests

class ThematicEncoder:
    def __init__(self):
        self.url = "http://localhost:11434/api/chat"
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": "llama3.3:70b",
            "format": "json",
            "options": {"temperature": 0.0},
            "stream": False,
        }
        self.system_message = (
            "You are an academic researcher analyzing opiate use related social media posts. Your task is to examine the language in posts and post titles, identify semantic themes, and classify the language based on specific rules related to context and thematic content. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
        )

    def encode(self, instructions, post, title, state_label):
        user_message = f"""
Instructions:

{instructions}

- Response Format:
  {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

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
    
    def write_prompt_structure(self, writer, feature_instruction):
        prompt = f"""
    data = {{
        {self.default_data.copy()},
    "messages": [
        {{"role:"system", "content"{self.system_message}}},
        {{"role": "user", "content": {{
    Instructions:

    {feature_instruction}

    - Response Format:
     {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

    - Respond based on the following inputs:
    Post: {{post}}
    Post Title: {{title}}
    State Label: {{state_label}}
        }}}},
    ]
    }}
    """
        writer.write(prompt)