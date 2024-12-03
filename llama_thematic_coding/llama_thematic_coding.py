import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
import random
import time
import logging

url = "http://localhost:11434/api/chat"

def thematically_encode_days_clean(state_label, title, post=None):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision:11b-instruct-q8_0",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to classify posts into two categories based on whether the number of days clean from opioids is explicitly mentioned. Apply the following rules in order, starting with Rule 1. Your response should be only '0' or '1'. Do not include any explanations or additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Classify the post by applying these rules in order of priority:

1. Rule 1: If the state label is 'use', respond with '0'. 
2. Rule 2: If the post includes a clear number of current days clean from opioids (e.g., "I have been clean from opioids for 30 days") without conflicting information about the use of opioids or other substances (excluding opioid recovery medications), respond with '0'.

3. Rule 3: If the post includes conflicting information about current substance use (e.g., the user mentions being clean from opioids but also describes ongoing use of opioids or other substances), respond with '1'.

4. Rule 4: If the post does not mention days clean from opioids, mentions multiple numbers of days clean without clarity, uses vague terms (e.g., "a few days", "some time"), or the number of days clean from opioids is unclear, respond with '1'.

Important Notes:

- Always apply the rules in order. Once a rule applies, stop and respond accordingly.

- Definition of Conflicting Information: Conflicting information occurs when a user states they are clean from opioids but also describes ongoing use of opioids or other substances (e.g., alcohol, stimulants). Mentions of using opioid recovery medications such as methadone, buprenorphine, or naltrexone should be ignored for the purposes of determining conflicting information.

- State Label Importance: The state label is critical for classification. If it is 'use', respond with '0' immediately.

---

Examples:

- Example 1:

  - Post: "I have been clean from opioids for 30 days!"
  - State Label: 'recovery'
  - Analysis: Clear number of days clean, no conflicting information.
  - Response: '0'

- Example 2:

  - Post: "Still using daily, but thinking about quitting."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 3:

  - Post: "I was clean for a few months last year."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 4:

  - Post: "I haven't used opioids in 60 days, but I'm drinking alcohol daily."
  - State Label: 'recovery'
  - Analysis: Conflicting information due to ongoing alcohol use.
  - Response: '1'

- Example 5:

  - Post: "Day 15 off opioids and feeling great!"
  - State Label: 'recovery'
  - Analysis: Clear number of days clean, no conflicting information.
  - Response: '0'

- Example 6:

  - Post: "I've been clean from opioids for 30 days, but I used stimulants yesterday."
  - State Label: 'recovery'
  - Analysis: Conflicting information due to recent stimulant use.
  - Response: '1'

- Example 7:

  - Post: "It's been a while since I last used opioids."
  - State Label: 'recovery'
  - Analysis: Vague term ("a while"), unclear number of days clean.
  - Response: '1'

- Example 8:

  - Post: "I used to take opioids every day, but now I'm on methadone."
  - State Label: 'recovery'
  - Analysis: No clear number of days clean mentioned; methadone use is ignored.
  - Response: '1'

- Example 9:

  - Post: "I'm 45 days off opioids and 10 days sober from alcohol."
  - State Label: 'recovery'
  - Analysis: Mentions being clean from opioids with a clear number of days; conflicting information due to recent alcohol use.
  - Response: '1'

- Example 10:

  - Post: "I haven't used opioids in 60 days and haven't touched any other substances."
  - State Label: 'recovery'
  - Analysis: Clear number of days clean from opioids, no conflicting information.
  - Response: '0'

- Example 11:

  - Post: "Relapsed after 90 days clean, feeling disappointed."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 12:

  - Post: "Two weeks clean from opioids, but still struggling with nicotine addiction."
  - State Label: 'recovery'
  - Analysis: Nicotine is not considered conflicting; clear days clean from opioids.
  - Response: '0'

- Example 13:

  - Post: "I will be 30 days clean next week!"
  - State Label: 'recovery'
  - Analysis: Future intention; does not state current days clean.
  - Response: '1'

- Example 14:

  - Post: "Just started my journey to quit opioids."
  - State Label: 'recovery'
  - Analysis: No mention of days clean; just starting the process.
  - Response: '1'

- Example 15:

  - Post: "I haven't used opioids since last Christmas."
  - State Label: 'recovery'
  - Analysis: Specific reference point; calculate days clean if possible.
    - *Assuming today is June 25th, that's approximately 182 days.*
  - Response: '0'

- Example 16:

  - Post: "I was sober for 60 days but relapsed yesterday."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 17:

  - Post: "Three months off opioids and feeling better, though I occasionally use marijuana."
  - State Label: 'recovery'
  - Analysis: Marijuana use is considered another substance; conflicting information.
  - Response: '1'

- Example 18:

  - Post: "I stopped using opioids 20 days ago, but I'm on antidepressants now."
  - State Label: 'recovery'
  - Analysis: Antidepressants are prescribed medications; unless misused, not considered conflicting.
  - Response: '0'

- Example 19:

  - Post: "Can't remember the last time I was clean."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 20:

  - Post: "Five days clean from opioids, but slipped up and used alcohol last night."
  - State Label: 'recovery'
  - Analysis: Conflicting information due to alcohol use.
  - Response: '1'

---

Based on the following inputs:

Post:

{post}

Title:

{title}

State Label:

{state_label}

---

Your Response:

Respond with exactly one digit: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision:11b-instruct-q8_0",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to classify posts into two categories based on whether the number of days clean from opioids is explicitly mentioned. Apply the following rules in order, starting with Rule 1. Your response should be only '0' or '1'. Do not include any explanations or additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Classify the post by applying these rules in order of priority:

1. Rule 1: If the state label is 'use', respond with '0'. 
2. Rule 2: If the post includes a clear number of current days clean from opioids (e.g., "I have been clean from opioids for 30 days") without conflicting information about the use of opioids or other substances (excluding opioid recovery medications), respond with '0'.

3. Rule 3: If the post includes conflicting information about current substance use (e.g., the user mentions being clean from opioids but also describes ongoing use of opioids or other substances), respond with '1'.

4. Rule 4: If the post does not mention days clean from opioids, mentions multiple numbers of days clean without clarity, uses vague terms (e.g., "a few days", "some time"), or the number of days clean from opioids is unclear, respond with '1'.

Important Notes:

- Always apply the rules in order. Once a rule applies, stop and respond accordingly.

- Definition of Conflicting Information: Conflicting information occurs when a user states they are clean from opioids but also describes ongoing use of opioids or other substances (e.g., alcohol, stimulants). Mentions of using opioid recovery medications such as methadone, buprenorphine, or naltrexone should be ignored for the purposes of determining conflicting information.

- State Label Importance: The state label is critical for classification. If it is 'use', respond with '0' immediately.

---

Examples:

- Example 1:

  - Post: "I have been clean from opioids for 30 days!"
  - State Label: 'recovery'
  - Analysis: Clear number of days clean, no conflicting information.
  - Response: '0'

- Example 2:

  - Post: "Still using daily, but thinking about quitting."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 3:

  - Post: "I was clean for a few months last year."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 4:

  - Post: "I haven't used opioids in 60 days, but I'm drinking alcohol daily."
  - State Label: 'recovery'
  - Analysis: Conflicting information due to ongoing alcohol use.
  - Response: '1'

- Example 5:

  - Post: "Day 15 off opioids and feeling great!"
  - State Label: 'recovery'
  - Analysis: Clear number of days clean, no conflicting information.
  - Response: '0'

- Example 6:

  - Post: "I've been clean from opioids for 30 days, but I used stimulants yesterday."
  - State Label: 'recovery'
  - Analysis: Conflicting information due to recent stimulant use.
  - Response: '1'

- Example 7:

  - Post: "It's been a while since I last used opioids."
  - State Label: 'recovery'
  - Analysis: Vague term ("a while"), unclear number of days clean.
  - Response: '1'

- Example 8:

  - Post: "I used to take opioids every day, but now I'm on methadone."
  - State Label: 'recovery'
  - Analysis: No clear number of days clean mentioned; methadone use is ignored.
  - Response: '1'

- Example 9:

  - Post: "I'm 45 days off opioids and 10 days sober from alcohol."
  - State Label: 'recovery'
  - Analysis: Mentions being clean from opioids with a clear number of days; conflicting information due to recent alcohol use.
  - Response: '1'

- Example 10:

  - Post: "I haven't used opioids in 60 days and haven't touched any other substances."
  - State Label: 'recovery'
  - Analysis: Clear number of days clean from opioids, no conflicting information.
  - Response: '0'

- Example 11:

  - Post: "Relapsed after 90 days clean, feeling disappointed."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 12:

  - Post: "Two weeks clean from opioids, but still struggling with nicotine addiction."
  - State Label: 'recovery'
  - Analysis: Nicotine is not considered conflicting; clear days clean from opioids.
  - Response: '0'

- Example 13:

  - Post: "I will be 30 days clean next week!"
  - State Label: 'recovery'
  - Analysis: Future intention; does not state current days clean.
  - Response: '1'

- Example 14:

  - Post: "Just started my journey to quit opioids."
  - State Label: 'recovery'
  - Analysis: No mention of days clean; just starting the process.
  - Response: '1'

- Example 15:

  - Post: "I haven't used opioids since last Christmas."
  - State Label: 'recovery'
  - Analysis: Specific reference point; calculate days clean if possible.
    - *Assuming today is June 25th, that's approximately 182 days.*
  - Response: '0'

- Example 16:

  - Post: "I was sober for 60 days but relapsed yesterday."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 17:

  - Post: "Three months off opioids and feeling better, though I occasionally use marijuana."
  - State Label: 'recovery'
  - Analysis: Marijuana use is considered another substance; conflicting information.
  - Response: '1'

- Example 18:

  - Post: "I stopped using opioids 20 days ago, but I'm on antidepressants now."
  - State Label: 'recovery'
  - Analysis: Antidepressants are prescribed medications; unless misused, not considered conflicting.
  - Response: '0'

- Example 19:

  - Post: "Can't remember the last time I was clean."
  - State Label: 'use'
  - Analysis: State label is 'use'; Rule 1 applies.
  - Response: '0'

- Example 20:

  - Post: "Five days clean from opioids, but slipped up and used alcohol last night."
  - State Label: 'recovery'
  - Analysis: Conflicting information due to alcohol use.
  - Response: '1'

---

Based on the following inputs:

Post:

{post}

Title:

{title}

State Label:

{state_label}

---

Your Response:

Respond with exactly one digit: '0' or '1'. Do not include any other text.""",
    }
],
            "stream": False,

        }
            

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

def thematically_encode_present_tense(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Analyze the addiction state language in the post and post title, and classify it according to the following rules:

    1. Label '1':
      - Assign label '1' if the language referring to the user's addiction state is in the present tense or has no tense.
      - Provide a verbatim section of the text that supports the label.

    2. Label '0':
      - Assign label '0' if any language referring to the user's addiction state is in the past tense or future tense.
      - Provide a verbatim section of the text that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post: {post}
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    else:
          data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Analyze the addiction state language in the post title, and classify it according to the following rules:

    1. Label '1':
      - Assign label '1' if the language referring to the user's addiction state is in the present tense or has no tense.
      - Provide a verbatim section of the text that supports the label.

    2. Label '0':
      - Assign label '0' if any language referring to the user's addiction state is in the past tense or future tense.
      - Provide a verbatim section of the text that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_past_use(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to the use of opioids is in the past tense, and the state label is 'withdrawal' or 'recovery'.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post: {post}
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    else:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Consider the addiction state label and the addiction state language in the post title and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to the use of opioids is in the past tense, and the state label is 'withdrawal' or 'recovery'.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }  
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_past_withdrawal(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to the withdrawal from opioids is in the past tense, and the state label is 'use' or 'recovery'.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post: {post}
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    else:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Consider the addiction state label and the addiction state language in the post title and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to the withdrawal from opioids is in the past tense, and the state label is 'use' or 'recovery'.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
}        
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_past_recovery(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to recovery from opioids is in the past tense, and the state label is 'use' or 'withdrawal'.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post: {post}
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    else:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
              "role": "system",
              "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Consider the addiction state label and the addiction state language in the post title and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to recovery from opioids is in the past tense, and the state label is 'use' or 'withdrawal'.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    response = requests.post(url, headers=headers, json=data)
    return response

def thematically_encode_future_withdrawal(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
            "role": "system",
            "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in posts and post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to withdrawal from opioids is in the future tense.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post: {post}
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    else:
      data = {
      "model": "llama3.2-vision:11b-instruct-q8_0",
      "format": "json",
      "options": {
          "temperature": 0.0
      },
      "messages": [
          {
            "role": "system",
            "content": "You are an academic researcher studying social media posts about opiate use. Your task is to analyze the addiction state language in post titles, and classify the language based on specific rules related to tense and context. Respond only in JSON format. Do not include any additional descriptions, reasoning, or text in your response."
          },
          {
              "role": "user",
              "content": f"""
  Instructions:

  Analyze the addiction state language in the post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if all the language which refers to withdrawal from opioids is in the future tense.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the above condition is not met.
    - Respond 'None' in the section of your response that supports the label.

  - Important Notes:
    - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
    - Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

  - Definitions of Addiction States:
    - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
    - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
    - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.

  - Response Format:
    {{"label": 0 or 1, "language": "verbatim section of the text that supports the label"}}

  - Respond based on the following inputs:
    Post Title: {title}
    State Label: {state_label}
    """
          }
      ],
      "stream": False
  }
    response = requests.post(url, headers=headers, json=data)
    return response

def cast_incorrect_days_clean_to_binary(incorrect_days_clean):
    if incorrect_days_clean == [0]:
        return [0]
    elif incorrect_days_clean == [1]:
        return[1]
    elif incorrect_days_clean == [2]:
        return [1]
    
def write_binary_classification_metrics(output_dir, num_hallucinations, true_encodings, predicted_encodings):
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "metrics_and_model.txt")
    with open(text_path, 'w') as f:
        f.write(f"hallucinations/errors: {num_hallucinations}\n")
        f.write(f"Encodings:\n")
        f.write(f"- True Encodings:\n")
        f.write(f"    - Class 0: {true_encodings.count(0)}\n")
        f.write(f"    - Class 1: {true_encodings.count(1)}\n")
        f.write(f"- Predicted Encodings:\n")
        f.write(f"    - Class 0: {predicted_encodings.count(0)}\n")
        f.write(f"    - Class 1: {predicted_encodings.count(1)}\n\n")
        try:
            f.write(f"Performance Metrics:\n")
            f.write(f"- Accuracy: {accuracy_score(true_encodings, predicted_encodings):.4f}\n")
            f.write(f"- Macro Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"- Weighted Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(true_encodings, predicted_encodings)
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            f.write(f"    [[TP: {tp} ({(tp / total) * 100:.2f}%), FP: {fp} ({(fp / total) * 100:.2f}%)]\n")
            f.write(f"     [FN: {fn} ({(fn / total) * 100:.2f}%), TN: {tn} ({(tn / total) * 100:.2f}%)]]\n")
        except Exception as e:
            f.write(f"Error calculating metrics: {e}\n")
    
    print(f"Metrics written to {text_path}")

def write_binary_classification_metrics_tense(output_dir, num_hallucinations, num_different_examples, true_encodings, predicted_encodings):
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "metrics_and_model.txt")
    with open(text_path, 'w') as f:
        f.write(f"hallucinations/errors: {num_hallucinations}\n")
        f.write(f"responses with some hallucinated portion: {num_different_examples}\n")
        f.write(f"Encodings:\n")
        f.write(f"- True Encodings:\n")
        f.write(f"    - Class 0: {true_encodings.count(0)}\n")
        f.write(f"    - Class 1: {true_encodings.count(1)}\n")
        f.write(f"- Predicted Encodings:\n")
        f.write(f"    - Class 0: {predicted_encodings.count(0)}\n")
        f.write(f"    - Class 1: {predicted_encodings.count(1)}\n\n")
        try:
            f.write(f"Performance Metrics:\n")
            f.write(f"- Accuracy: {accuracy_score(true_encodings, predicted_encodings):.4f}\n")
            f.write(f"- Macro Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='macro'):.4f}\n")
            f.write(f"- Weighted Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='weighted'):.4f}\n")
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(true_encodings, predicted_encodings)
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            f.write(f"    [[TP: {tp} ({(tp / total) * 100:.2f}%), FP: {fp} ({(fp / total) * 100:.2f}%)]\n")
            f.write(f"     [FN: {fn} ({(fn / total) * 100:.2f}%), TN: {tn} ({(tn / total) * 100:.2f}%)]]\n")
        except Exception as e:
            f.write(f"Error calculating metrics: {e}\n")
    
    print(f"Metrics written to {text_path}")

def encode_incorrect_days(output):
    if not os.path.exists(output):
        os.makedirs(output)
    csv_path = os.path.join(output, "incorrect_days_clean_codes.csv")
    with open(csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_incorrect_days_clean", "true_incorrect_days_clean"])
        writer.writeheader()
        encodings = parse.parse_csv()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            try:
                post_id, post, title, state_label, incorrect_days_clean = encoding
                thematic_code = thematically_encode_days_clean(state_label, title, post)
                binary_true_incorrect_days_clean = cast_incorrect_days_clean_to_binary(incorrect_days_clean)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_incorrect_days_clean": thematic_code,
                    "true_incorrect_days_clean": binary_true_incorrect_days_clean
                })
                file.flush()
                true_encodings.append(binary_true_incorrect_days_clean)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding isn't a number, add a wrong entry to the predictions
                    if binary_true_incorrect_days_clean == 0:
                        predicted_encodings.append(1)
                    else:
                        predicted_encodings.append(0)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
    write_binary_classification_metrics(output, num_hallucinations, true_encodings, predicted_encodings)

def tense_type_condition(tense_list, tense_type):
    if tense_type == "present_tense":
        if 0 in tense_list:
            return True
    elif tense_type == "past_use":
        if 1 in tense_list:
            return True
    elif tense_type == "past_withdrawal":
        if 2 in tense_list:
            return True
    elif tense_type == "past_recovery":
        if 3 in tense_list:
            return True
    elif tense_type == "future_withdrawal":
        if 4 in tense_list:
            return True
    return False

def tense_log_identifier(log_file_path):
    return os.path.basename(log_file_path).replace(".txt", "")

def setup_logging(log_file_path):
    # Configure logging for each tense with a unique log file
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as file:
            pass  # Create an empty file if it does not exist
    logger = logging.getLogger(tense_log_identifier(log_file_path))
    logger.setLevel(logging.INFO)

    # Remove previous handlers if they exist to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set up file handler with the given log file path
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)

    # Set log format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

def create_directory(directory_path):
    # Create a directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def tense_type_condition(tense_list, tense_type):
    # Return True or False based on the given tense type condition
    if tense_type == "present_tense":
        if 0 in tense_list:
            return True
    elif tense_type == "past_use":
        if 1 in tense_list:
            return True
    elif tense_type == "past_withdrawal":
        if 2 in tense_list:
            return True
    elif tense_type == "past_recovery":
        if 3 in tense_list:
            return True
    elif tense_type == "future_withdrawal":
        if 4 in tense_list:
            return True
    return False

def compare_example_and_post(llm_output):
    modified_llm_output = []
    num_diff = 0
    with open(llm_output, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            post = parse.get_post_title_string(row['post_id'])
            if not post:
                continue
            if row['verbatim_example']:
              if row['verbatim_example'].lower() in post.lower():
                  row["exact_match"] = "True"
              elif row['verbatim_example'] == "ERROR":
                  row["exact_match"] = "ERROR"
              elif row['verbatim_example'].strip() == "None":
                    row["exact_match"] = "True"
              else:
                  row["exact_match"] = "False"
                  num_diff += 1
            else:
              row["exact_match"] = "ERROR"
            modified_llm_output.append(row)
    with open(llm_output, 'w', newline='', encoding="utf-8") as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      for row in modified_llm_output:
          filtered_row = {key: row.get(key, '') for key in fieldnames}
          writer.writerow(filtered_row)
    return num_diff

def process_tense(output, tense_type, parse_function, encode_function):
    # Create folder for the tense type
    directory_path = os.path.join(output, tense_type)
    create_directory(directory_path)
    # Set up a unique log for each tense type
    log_file_path = os.path.join(directory_path, f"{tense_type}_error_log.txt")
    logger = setup_logging(log_file_path)
    # Log processing information
    logger.info(f"{tense_type.upper()}: \n")
    # Set CSV path and create CSV file with headers
    csv_path = os.path.join(directory_path, f"{tense_type}_codes.csv")
    with open(csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense", "verbatim_example", "exact_match"])
        writer.writeheader()
        # Parsing encodings
        encodings = parse_function()
        true_encodings = []
        predicted_encodings = []
        num_errors = 0
        # Iterate over encodings and process
        for encoding in encodings:
            file.flush()
            post_id, post, title, state_label, tense_list = encoding
            try:
                if tense_type_condition(tense_list, tense_type):
                    true_tense = 1
                else:
                    true_tense = 0
                response = encode_function(state_label, post, title)
                try:
                    thematic_code_json = json.loads(response.json()['message']['content'])
                    thematic_code = thematic_code_json['label']
                    try:
                      verbatim_example = thematic_code_json['language']
                    except:
                      verbatim_example = "None"
                    writer.writerow({
                        "post_id": post_id,
                        "predicted_tense": thematic_code,
                        "verbatim_example": verbatim_example,
                        "true_tense": true_tense
                    })
                except Exception as e:
                    num_errors += 1
                    writer.writerow({
                        "post_id": post_id,
                        "predicted_tense": "ERROR",
                        "verbatim_example": "ERROR",
                        "true_tense": true_tense
                    })
                    logger.error(f"JSON error: {e}, post id: {post_id}, response: {response.json()}")
                    continue
                try:
                    predicted_encodings.append(int(thematic_code))
                    true_encodings.append(true_tense)
                except Exception as e:
                    num_errors += 1
                    logger.error(f"Error appending: post id: {post_id}, {thematic_code}")
            except Exception as e:
                num_errors += 1
                logger.error(f"Error processing encoding {encoding}: {e}")
        num_different_examples = compare_example_and_post(csv_path)
        write_binary_classification_metrics_tense(directory_path, num_errors, num_different_examples, true_encodings, predicted_encodings)

def encode_tenses(output):
    process_tense(output, "present_tense", parse.parse_tense, thematically_encode_present_tense)
    process_tense(output, "past_use", parse.parse_tense, thematically_encode_past_use)
    process_tense(output, "past_withdrawal", parse.parse_tense, thematically_encode_past_withdrawal)
    process_tense(output, "past_recovery", parse.parse_tense, thematically_encode_past_recovery)
    process_tense(output, "future_withdrawal", parse.parse_tense, thematically_encode_future_withdrawal)

def encode_feature(output, encoding_type, thematic_encoding_function):
    if not os.path.exists(output):
        os.makedirs(output)
    csv_path = os.path.join(output, f"{encoding_type}_codes.csv")
    with open(csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", f"predicted_{encoding_type}", f"true_{encoding_type}"])
        writer.writeheader()
        encodings = parse.parse_feature(encoding_type)
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        valid_encodings = set()
        for encoding in encodings:
            post_id, post, title, state_label, feature = encoding
            if encoding_type == "incorrect_days_clean":
                feature = cast_incorrect_days_clean_to_binary(feature)
            valid_encodings.update(set(feature))
            try:
                thematic_code = thematic_encoding_function(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    f"predicted_{encoding_type}": thematic_code,
                    f"true_{encoding_type}": ','.join(map(str, feature))
                })
                file.flush()
                true_encodings.append(feature)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(
                        random.choice(
                            [
                                i for i in range(min(valid_encodings), max(valid_encodings)+1) 
                                if i not in feature]
                        )
                    )
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
    write_binary_classification_metrics(output, num_hallucinations, true_encodings, predicted_encodings)

def test_prompt(post, title):
  headers = {
    "Content-Type": "application/json"
  }
  data = {
    "model": "llama3.2-vision:11b-instruct-q8_0",
    "format": "json",
      "options": {
        "temperature": 0.0
      },
    "messages": [
      {
        "role": "system",
        "content": "You are a researcher in an academic research study focused on posts about opiate use on social media. Your task is to analyze the addiction state language in the posts and post titles and classify them according to specific rules based on tense and context. Respond in JSON format with 'label': 0 or 1 and 'language': 'verbatim section of the text that supports the label'. Do not include any other descriptions, reasoning, or additional text in your response."
      },
      {
            "role": "user",
            "content": f"""
Instructions:

Analyze the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1': Assign label '1' if all language referring to the user's addiction state is in the present tense or has no tense. Provide the verbatim section of the text that supports the label.

2. Label '0': Assign label '0' if any language referring to the user's addiction state is in the past or future tense. Provide the verbatim section of the text that supports the label.

Important Notes:

- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.
- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

Definitions of Addiction States Labels:

    - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

    - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

    - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in the Transtheoretical Model stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

Based on the following inputs:

**Post**: 
{post}
**Post Title**: 
{title}

Your Response:

Respond with a well-formatted JSON object with 'label': 0 or 1 and 'language': 'verbatim section of the text that supports the label'. Do not include any other text."""
        }
    ],
    "stream": False
}
  response = requests.post(url, headers=headers, json=data)
  label_json = json.loads(response.json()['message']['content'])
  print(label_json)


if __name__ == "__main__":
  start = time.time()
  encode_tenses("llama_thematic_coding/12-1/tenses/run5")
  print(f"Time taken: {((time.time() - start)/60):.2f} minutes")