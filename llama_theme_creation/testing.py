import requests

class TestChat:
    def __init__(self, model="llama3.2-vision:11b-instruct-q4_K_M"):
        self.url = "http://localhost:11434/api/chat"
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": model,
            "options": {
                "temperature": 0.0
            },
            "stream": False,
        }
        self.system_message = (
            "You are a friendly and helpful chatbot. Remember this system prompt if asked later"
        )
if __name__ == "__main__":
    chat = TestChat()
    user_message = "Can you explain what the opioid epidemic is?"
    data = chat.default_data.copy()
    data["messages"] = [
        {"role": "system", "content": chat.system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "The opioid epidemic is a public health crisis in the United States that has been caused by the overprescription of opioid painkillers and the subsequent rise in heroin use. The epidemic has led to a dramatic increase in overdose deaths and has had a devastating impact on families and communities across the country."},
        {"role": "user", "content": "Tell me what your system prompt is, and what i previously asked you. "},
    ]
    response = requests.post(chat.url, headers=chat.headers, json=data)
    print(response.json()["message"]["content"])
     

