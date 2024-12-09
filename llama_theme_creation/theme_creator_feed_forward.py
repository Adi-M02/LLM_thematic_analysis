import requests

class ThemeCreatorFeedForward:
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
            "You are an academic researcher analyzing the themes related to opiate addiction state characterization on social media. You will be given a post, post title and a list of existing themes you previously found among related posts. Your task is to analyze the post and identify major themes related to opiate addiction state characterization in it. If a theme you identify is not in the list of existing themes add it to the list of themes. If a new theme is similar to a previous theme combine them into a new more general theme. Only create new themes if you are sure the posts can't be classified with an existing theme. Respond only in the specified format with the major themes related to opiate addiction state characterization. Do not include any additional descriptions, reasoning, or text in your response."
        )
    def create_themes(self, post, title):
        user_message = f"""
Instructions:

Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only by appending new major themes related to opiate addiction state classification to the input list. Only append if the new theme is not similar to an existing theme. If a new theme is similar to an existing theme combine the themes into a new more general theme. If no new themes are identified return the input list of themes unmodified. Do not include any additional descriptions, reasoning, or text in your response.

- Important Notes:
  - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
{{
  "themes": [
    "Title of theme", "Title of another theme", ...
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
    
    def write_prompt_structure(self, writer):
      prompt = f"""
        data = {{
            {self.default_data.copy()},
        "messages": [
            {{"role:"system", "content"{self.system_message}}},
            {{"role": "user", "content": 
              Instructions:

              Analyze the opiate addiction state information in the post and post title and identify the major theme or themes related to opiate addiction state characterization in the text. Respond only by appending new major themes related to opiate addiction state classification to the input list. Only append if the new theme is not similar to an existing theme. If a new theme is similar to an existing theme combine the themes into a new more general theme. If no new themes are identified return the input list of themes unmodified. Do not include any additional descriptions, reasoning, or text in your response.

              - Important Notes:
                - Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
              - Definitions of Addiction States:
                - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
                - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
                - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

              - Response Format:
              {{
                "themes": [
                  "Title of theme","Title of another theme", ...
                ]
              }}
                
              - Respond based on the following inputs:
                Post: {{post}}
                Post Title: {{title}}
              }}
            }}
    """
      writer.write(prompt)
if __name__ == "__main__":
  pass