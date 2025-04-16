from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-90B-Vision-Instruct")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-90B-Vision-Instruct")

text_prompt = "Describe thematic coding for social media posts related to opiate use"

inputs = processor(text=text_prompt, return_tensors="pt")

# Generate a response
outputs = model.generate(
    **inputs,
    max_length=200,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Decode the output
response = processor.decode(outputs[0], skip_special_tokens=True)

# Print the model's response
print("Model's response:", response)