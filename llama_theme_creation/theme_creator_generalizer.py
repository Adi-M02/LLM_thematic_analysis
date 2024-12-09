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
                    "type": "string"
                  }
                }
              },
              "required": ["themes"]
            }
        }
        self.system_message = (
            "You are an academic researcher analyzing the themes related to opiate addiction state characterization on social media. You will be given a post, post title and a list of existing themes you previously found among related posts. Your task is to analyze the post and identify major themes related to opiate addiction state characterization in it. "
        )
    def create_themes(self, post, title, themes):
        user_message = f"""