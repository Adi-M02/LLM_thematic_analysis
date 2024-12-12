import requests

class ThemeCreatorGeneralizer:
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
            "You are an academic researcher analyzing the themes related to opiate addiction state characterization in a dataset of posts related to opiate use from social media. Your task is to find the 5 most significant themes describing opiate addiction state characterization in the entire dataset. You will be given a post, post title, a list of themes which were found to apply to individual posts in the dataset, and the previous list of major themes that have been found in the dataset."
        )
    def generalize_themes(self, post, title, themes, major_themes):
        user_message = f"""
Instructions:
Analyze the opiate addiction state information you see in the post and post title and identify major themes related to opiate addiction state characterization in the text. Use your knowledge of addiction state characterization, the content of the post text, the list of themes present in all posts in the dataset, and the previously generated major themes to identify and give a description of 5 major themes in the dataset. The 5 themes should be the most significant themes that describe opiate addiction state characterization in the dataset. Remember that you are only seeing one post in the entire dataset so do not include specific information from the post in any of the 5 themes or their descriptions. Respond only in the specified format with the major themes related to opiate addiction state characterization and a description of each theme. Do not include any additional descriptions, reasoning, or text in your response.

- Important Notes:
- Addiction state language refers to any mentions of use, withdrawal, or recovery related to opiate addiction.
- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

- Response Format:
  "themes": [
    {{
      "theme": "Title of theme",
      "description": "Description of theme"
    }},
    {{
      "theme": "Title of another theme",
      "description": "Description of another theme"
    }},
    ...
    ]

- Respond based on the following inputs:
    Post: {post}
    Post Title: {title}
    Themes: {themes}
    Major Themes: {major_themes}
    """
        data = self.default_data.copy()
        data["messages"] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        response = requests.post(self.url, headers=self.headers, json=data)
        return response