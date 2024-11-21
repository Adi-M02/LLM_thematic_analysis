import requests
import json

url = "http://localhost:11434/api/chat"

# def llama3(prompt):
#     data = {
#         "model": "llama3.2-vision",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "Your role is to understand the cause and effect relationship in social media posts."
#             },
#             {
#                 "role": "user",
#                 "content": "Based on your role, can you explain the term 'causal gist' in relation to sentences that have causal coherence?"
#             },
#             {
#                 "role": "user",
#                 "content": f"So based on the sentences: 'So tomorrow is the big day, where I start recovery at solutions in Las Vegas. Looks like a 30 to 45 day inpatient rehab, with an emphasis on pain management and depression/anxiety.I've been using oxycodone for about seven years now to the point where my daily average is about 250 mg per day, which has managed to allow me to lose my family and my parents trust. And that means it's time to get better.Wish me luck! I don't think that I can use digital device is well there, but I've watched many of you get better, and I know the feeling of seeing success, and it's good. I hope that I have the same success!' Is there a cause-effect relationship in this given sentences?- If yes, answer only: 'Yes'- If no, just answer: 'No'- Don’t give me any explanations"
#             },
#             {"role": "user",
#             "content": "Indeed, there is a cause-effect relationship in the given sentences. Extract the corresponding cause phrase and effect phrase in the given sentences. Just respond in JSON format: {”Cause”: “”, “Effect”:""}"
#             },
#             {
#                 "role": "user",
#                 "content": "Generate a reasonable and clear one sentence causal gist based on {'Cause': 'Using oxycodone for about seven years now', 'Effect':'managed to allow me to lose my family and my parents trust'} and your understanding of the sentence with the cause-effect relationship."
#             }
#         ],
#         "stream": False,
#     }

#     headers = {
#         "Content-Type": "application/json"
#     }

#     response = requests.post(url, headers=headers, json=data)
#     return response.json()['message']['content']

# print(llama3("I took the vaccine yesterday. I'm really sick now."))

def test_prompt():
    data = {
        "model": "llama3.2-vision-0temp",
        "messages": [
            {
                "role": "system",
                "content": "You answer all questions truthfully"
            },
            {
                "role": "user",
                "content": "What is the temperature parameter of your model set to currently?"
            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

if __name__ == "__main__":
    print(test_prompt())