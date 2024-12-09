import requests

class ThemeCreator:
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
                                },
                                "example": {
                                    "type": "string"
                                }
                            },
                            "required": ["theme", "description", "example"]
                        }
                    }
                },
                "required": ["themes"]
            }
        }
        self.system_message = (
            "You are an academic researcher analyzing the themes related to opiate addiction state characterization on social media. Your task is to analyze a post and understand its thematic content. Respond only in the specified format with the major themes related to opiate addiction state characterization, description of the theme, and verbatim example showing support of the theme. Do not include any additional descriptions, reasoning, or text in your response."
        )
    def create_themes(self, post, title):
        user_message = f"""
Instructions:

Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization you see. Respond only in the specified format with the major theme or themes related to opiate addiction state characterization, the description of each theme, and a verbatim example from the post that supports the the theme being in the text. Do not include any additional descriptions, reasoning, or text in your response.

- Important Notes:
  - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
{{
  "themes": [
    {{
      "theme": "Title of theme",
      "description": "Definition of the theme", 
      "example": "Verbatim example from the post"
    }},
    {{
      "theme": "Title of another theme",
      "description": "Definition of the theme", 
      "example": "Verbatim example from the post"
    }}, 
  ]
}}
If multiple major themes related to the user's opiate addiction state characterization are present, include all themes in the response. If no themes are present, respond with an empty array.
  
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
    
    def write_prompt_structure(self, writer):
      prompt = f"""
        data = {{
            {self.default_data.copy()},
        "messages": [
            {{"role:"system", "content"{self.system_message}}},
            {{"role": "user", "content": 
              Instructions:

              Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization you see. Respond only in the specified format with the major theme or themes related to opiate addiction state characterization, the description of each theme, and a verbatim example from the post that supports the the theme being in the text. Do not include any additional descriptions, reasoning, or text in your response.

              - Important Notes:
                - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
              - Definitions of Addiction States:
                - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
                - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
                - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

              - Response Format:
              {{
                "themes": [
                  {{
                    "theme": "Title of theme",
                    "description": "Definition of the theme", 
                    "example": "Verbatim example from the post"
                  }},
                  {{
                    "theme": "Title of another theme",
                    "description": "Definition of the theme", 
                    "example": "Verbatim example from the post"
                  }}, 
                ]
              }}
              If multiple major themes related to the user's opiate addiction state characterization are present, include all themes in the response. If no themes are present, respond with an empty array.
                
              - Respond based on the following inputs:
                Post: {{post}}
                Post Title: {{title}}
              }}
            }}
    """
      writer.write(prompt)
if __name__ == "__main__":
    creator = ThemeCreator()
    writer = ""
    creator.write_prompt_structure("test.txt")