import anthropic


client = anthropic.Anthropic(api_key='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')

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=0,
    system="You are a researcher in a study on opiate use on reddit",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Explain what thematic encoding to understand opiate use in social media is"
                }
            ]
        }
    ]
)
print(message.content)


