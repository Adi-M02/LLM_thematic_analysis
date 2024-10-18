import openai

system_message = {"role": "system", "content": "Your role is to understand the cause and effect relationship in social media posts."}

# First API call
response1 = openai.chat.completions.create(
    model="gpt-4-turbo",  # Ensure correct model name
    messages=[
        system_message,
        {"role": "user", "content": "Can you provide a brief definition of what a cause-effect relationship is?"}
    ]
)
print(response1.choices[0].message.content)

# Second API call
response2 = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        system_message,
        {"role": "user", "content": "Based on your role, can you explain the term 'causal gist' in relation to sentences that have causal coherence?"}
    ]
)
print(response2.choices[0].message.content)

# Third API call
response3 = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        system_message,
        {"role": "user", "content": "So based on the sentence: 'I took the vaccine yesterday. I'm really sick now.' Is there a cause-effect relationship in this given sentence?- If yes, just answer: 'Yes'- If no, just answer: 'No'- Don’t give me any explanations"}
    ]
)
print(response3.choices[0].message.content)

response4 = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        system_message,
        {"role": "user", "content": "Generate a reasonable and clear causal gist based on {'Cause': 'took the vaccine', 'Effect':'really sick now} and your understanding of the sentence with the cause-effect relationship."
        }
    ]
)
print(response4.choices[0].message.content)