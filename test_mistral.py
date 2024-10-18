# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# def get_current_weather(location: str, format: str):
#     """
#     Get the current weather

#     Args:
#         location: The city and state, e.g. San Francisco, CA
#         format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
#     """
#     pass

# conversation = [{"role": "user", "content": "What's the weather like in Paris?"}]
# tools = [get_current_weather]


# # format and tokenize the tool use prompt 
# inputs = tokenizer.apply_chat_template(
#             conversation,
#             tools=tools,
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt",
# )

# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# inputs.to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=1000)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('HF_TOKEN')

client = InferenceClient(api_key=os.getenv('HF_TOKEN'))
for message in client.chat_completion(
	model="mistralai/Mistral-7B-Instruct-v0.3",
	messages=[
        {"role": "system", 
        "content": "Your role is to understand the cause and effect relationship in social media posts."
        },
        {"role": "user",
        "content": "Based on your role, can you explain the term 'causal gist' in relation to sentences that have causal coherence?"
        },
        {"role": "assistant",
         "content":"None"},
        {"role": "user",
        "content": f"So based on the sentences: 'So tomorrow is the big day, where I start recovery at solutions in Las Vegas. Looks like a 30 to 45 day inpatient rehab, with an emphasis on pain management and depression/anxiety.I've been using oxycodone for about seven years now to the point where my daily average is about 250 mg per day, which has managed to allow me to lose my family and my parents trust. And that means it's time to get better.Wish me luck! I don't think that I can use digital device is well there, but I've watched many of you get better, and I know the feeling of seeing success, and it's good. I hope that I have the same success!' Is there a cause-effect relationship in this given sentence?- If yes, answer only: 'Yes'- If no, just answer: 'No'- Don’t give me any explanations"
        }, 
        {"role": "assistant",
         "content":"None"},
        {"role": "user",
        "content": "Indeed, there is a cause-effect relationship in the given sentences. Extract the corresponding cause phrase and effect phrase in the given sentences. Just respond in JSON format: {”Cause”: “”, “Effect”:""}"
        }, 
        {"role": "assistant",
          "content":"None"},
        {"role": "user",
        "content": "Generate a reasonable and clear one sentence causal gist based on {'Cause': 'Using oxycodone for about seven years, losing family and parents' trust', 'Effect':'Time to get better'} and your understanding of the sentence with the cause-effect relationship."
        }
           ],
	max_tokens=500,
	stream=True,
):
  print(message.choices[0].delta.content, end="")