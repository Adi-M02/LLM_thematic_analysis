import requests

class ThemeCreatorNoFeedForward:
    def __init__(self, model="llama3.2-vision:11b-instruct-q4_K_M"):
        self.url = "http://localhost:11434/api/chat"
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": model,
            "options": {
                "temperature": 0.0
            },
            "stream": False,
            "format": {
              "type": "object",
              "properties": {
                "themes": {
                  "type": "array",
                  "items": {
                    "type": "string"
                  }
                }
              },
              "required": ["themes"]
            }
        }
        self.system_message = (
            "You are an academic researcher analyzing the themes related to opiate addiction state characterization on social media. You will be given a post and post title. Your task is to analyze the post and identify major themes related to opiate addiction state characterization in it. Only include themes which you know from your understanding of opiate addiction can be used to characterize the post author's opiate addiction state.  Respond only in the specified format with the major themes related to opiate addiction state characterization. Do not include any additional descriptions, reasoning, or text in your response."
        )
    def create_themes(self, post, title):
        user_message = f"""
Instructions:

Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only with the themes related to the author's opiate addiction state characterization. Do not include any additional descriptions, reasoning, or text in your response.

- Important Notes:
  - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  "themes": [
    "Title of theme", "Title of another theme", ...
  ]
  
- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
"""
        
        data = self.default_data.copy()
        data["messages"] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        response = requests.post(self.url, headers=self.headers, json=data)
        return response
    
class ThemeCreatorNoFeedForwardWDesc:
    def __init__(self, model="llama3.2-vision:11b-instruct-q4_K_M"):
        self.url = "http://localhost:11434/api/chat"
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": model,
            "options": {
                "temperature": 0.0
            },
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme": {
                                    "type": "string"
                                },
                                "description": {
                                    "type": "string"
                                }
                            },
                            "required": ["theme", "description"]
                        }
                    }
                },
                "required": ["themes"]
            },
        }
        self.system_message = (
            "You are an academic researcher analyzing the themes related to opiate addiction state characterization on social media. You will be given a post and post title. Your task is to analyze the post and identify major themes and their descriptions related to opiate addiction state characterization. Only include themes which you know from your understanding of opiate addiction can be used to characterize the post author's opiate addiction state.  Respond only in the specified format with the major themes and the description of the theme related to opiate addiction state characterization. Do not include any additional descriptions, reasoning, or text in your response."
        )
    def create_themes(self, post, title):
        user_message = f"""
Instructions:

Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only with the themes and descriptions related to the author's opiate addiction state characterization. Do not include any additional descriptions, reasoning, or text in your response.

- Important Notes:
  - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  {{
    "themes": [
      {{"theme": "Title of theme", "description": "Description of theme"}}, {{"theme": "Title of another theme", "description": "Description of another theme"}}, ...
    ]
  }}
  
- Respond based on the following inputs:
  Post: {post}
  Post Title: {title}
"""
        
        data = self.default_data.copy()
        data["messages"] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        response = requests.post(self.url, headers=self.headers, json=data)
        return response