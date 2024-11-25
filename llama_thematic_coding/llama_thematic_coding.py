import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os
import random

url = "http://localhost:11434/api/chat"

def thematically_encode_days_clean(state_label, title, post=None):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision-0temp",
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
            "model": "llama3.2-vision-0temp",
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
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic research study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the user's addiction state is not in the past or future tense respond with '1'.

2. Code '1': If there is language which refers to the user's addiction state in the present tense respond with '1'.

3. Code '0': Otherwise respond with '0'.


Important Notes:

- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in the Transtheoretical Model (TTM) stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

Examples:

- Example 1:

  - Post Title: "Struggling with addiction every day"
  - Addiction State Language: "Struggling with addiction" (present tense)
  - Classification: '1'

- Example 2:

  - Post Title: "I overcame my addiction last year"
  - Addiction State Language: "Overcame my addiction" (past tense)
  - Classification: '0'

---

Based on the following inputs:

Post Title:

{title}

Post:

{post}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic research study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the user's addiction state is in present tense, respond with '1'.

2. Code '0': If there is any language which refers to the user's addiction state that is not in present tense, respond with '0'.

Important Notes:

- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in the Transtheoretical Model (TTM) stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

Examples:

- Example 1:

  - Post Title: "Struggling with addiction every day"
  - Addiction State Language: "Struggling with addiction" (present tense)
  - Classification: '1'

- Example 2:

  - Post Title: "I overcame my addiction last year"
  - Addiction State Language: "Overcame my addiction" (past tense)
  - Classification: '0'

---

Based on the following inputs:

Post Title:

{title}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

def thematically_encode_past_use(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the use of opioids is in the past tense, and the state label is 'withdrawal' or 'recovery', respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I used opioids for years but now I'm clean"

  - Addiction State Language: "used opioids" (past tense)

  - State Label: 'recovery'

  - Classification: '1'

- Example 2:

  - Post Title: "Currently struggling with opioid addiction"

  - Addiction State Language: "struggling with opioid addiction" (present tense)

  - State Label: 'use'

  - Classification: '0'

- Example 3:

  - Post Title: "Will start rehab next week"

  - Addiction State Language: "Will start rehab" (future tense)

  - State Label: 'withdrawal'

  - Classification: '0'

- Example 4:

  - Post Title: "Overcame my addiction last year"

  - Addiction State Language: "Overcame my addiction" (past tense)

  - State Label: 'recovery'

  - Classification: '1'

- Example 5:

  - Post Title: "I am using again after relapse"

  - Addiction State Language: "am using again" (present tense)

  - State Label: 'use'

  - Classification: '0'

---

Based on the following inputs:

Post Title:

{title}

Post:

{post}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the use of opioids is in the past tense, and the state label is 'withdrawal' or 'recovery', respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I used opioids for years but now I'm clean"

  - Addiction State Language: "used opioids" (past tense)

  - State Label: 'recovery'

  - Classification: '1'

- Example 2:

  - Post Title: "Currently struggling with opioid addiction"

  - Addiction State Language: "struggling with opioid addiction" (present tense)

  - State Label: 'use'

  - Classification: '0'

- Example 3:

  - Post Title: "Will start rehab next week"

  - Addiction State Language: "Will start rehab" (future tense)

  - State Label: 'withdrawal'

  - Classification: '0'

- Example 4:

  - Post Title: "Overcame my addiction last year"

  - Addiction State Language: "Overcame my addiction" (past tense)

  - State Label: 'recovery'

  - Classification: '1'

- Example 5:

  - Post Title: "I am using again after relapse"

  - Addiction State Language: "am using again" (present tense)

  - State Label: 'use'

  - Classification: '0'

---

Based on the following inputs:

Post Title:

{title}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

def thematically_encode_past_withdrawal(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the withdrawal from opioids is in the past tense, and the state label is 'use' or 'recovery', respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I went through withdrawal last month but now I'm using again"

  - Addiction State Language: "went through withdrawal" (past tense, referring to withdrawal)

  - State Label: 'use'

  - Classification: '1'

- Example 2:

  - Post Title: "Currently experiencing withdrawal symptoms"

  - Addiction State Language: "experiencing withdrawal symptoms" (present tense, referring to withdrawal)

  - State Label: 'withdrawal'

  - Classification: '0'

- Example 3:

  - Post Title: "Just finished withdrawal and starting recovery"

  - Addiction State Language: "finished withdrawal" (past tense, referring to withdrawal)

  - State Label: 'recovery'

  - Classification: '1'

- Example 4:

  - Post Title: "Will begin withdrawal process tomorrow"

  - Addiction State Language: "Will begin withdrawal process" (future tense, referring to withdrawal)

  - State Label: 'use'

  - Classification: '0'

- Example 5:

  - Post Title: "Struggling with withdrawal symptoms today"

  - Addiction State Language: "Struggling with withdrawal symptoms" (present tense, referring to withdrawal)

  - State Label: 'withdrawal'

  - Classification: '0'

---

Based on the following inputs:

Post Title:

{title}

Post:

{post}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the withdrawal from opioids is in the past tense, and the state label is 'use' or 'recovery', respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I went through withdrawal last month but now I'm using again"

  - Addiction State Language: "went through withdrawal" (past tense, referring to withdrawal)

  - State Label: 'use'

  - Classification: '1'

- Example 2:

  - Post Title: "Currently experiencing withdrawal symptoms"

  - Addiction State Language: "experiencing withdrawal symptoms" (present tense, referring to withdrawal)

  - State Label: 'withdrawal'

  - Classification: '0'

- Example 3:

  - Post Title: "Just finished withdrawal and starting recovery"

  - Addiction State Language: "finished withdrawal" (past tense, referring to withdrawal)

  - State Label: 'recovery'

  - Classification: '1'

- Example 4:

  - Post Title: "Will begin withdrawal process tomorrow"

  - Addiction State Language: "Will begin withdrawal process" (future tense, referring to withdrawal)

  - State Label: 'use'

  - Classification: '0'

- Example 5:

  - Post Title: "Struggling with withdrawal symptoms today"

  - Addiction State Language: "Struggling with withdrawal symptoms" (present tense, referring to withdrawal)

  - State Label: 'withdrawal'

  - Classification: '0'

---

Based on the following inputs:

Post Title:

{title}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

def thematically_encode_past_recovery(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the recovery from opioids is in the past tense, and the state label is 'use' or 'withdrawal', respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I recovered from addiction last year"

  - Addiction State Language: "recovered from addiction" (past tense, referring to recovery)

  - State Label: 'use'

  - Classification: '1'

- Example 2:

  - Post Title: "Currently in recovery and feeling hopeful"

  - Addiction State Language: "in recovery and feeling hopeful" (present tense, referring to recovery)

  - State Label: 'recovery'

  - Classification: '0'

- Example 3:

  - Post Title: "I was sober for six months before relapse"

  - Addiction State Language: "I was sober for six months" (past tense, referring to recovery)

  - State Label: 'withdrawal'

  - Classification: '1'

- Example 4:

  - Post Title: "Hoping to start recovery soon"

  - Addiction State Language: "Hoping to start recovery" (future tense, referring to recovery)

  - State Label: 'use'

  - Classification: '0'

- Example 5:

  - Post Title: "Completed recovery but now back to using"

  - Addiction State Language: "Completed recovery" (past tense, referring to recovery)

  - State Label: 'use'

  - Classification: '1'

---

Based on the following inputs:

Post Title:

{title}

Post:

{post}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the recovery from opioids is in the past tense, and the state label is 'use' or 'withdrawal', respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I recovered from addiction last year"

  - Addiction State Language: "recovered from addiction" (past tense, referring to recovery)

  - State Label: 'use'

  - Classification: '1'

- Example 2:

  - Post Title: "Currently in recovery and feeling hopeful"

  - Addiction State Language: "in recovery and feeling hopeful" (present tense, referring to recovery)

  - State Label: 'recovery'

  - Classification: '0'

- Example 3:

  - Post Title: "I was sober for six months before relapse"

  - Addiction State Language: "I was sober for six months" (past tense, referring to recovery)

  - State Label: 'withdrawal'

  - Classification: '1'

- Example 4:

  - Post Title: "Hoping to start recovery soon"

  - Addiction State Language: "Hoping to start recovery" (future tense, referring to recovery)

  - State Label: 'use'

  - Classification: '0'

- Example 5:

  - Post Title: "Completed recovery but now back to using"

  - Addiction State Language: "Completed recovery" (past tense, referring to recovery)

  - State Label: 'use'

  - Classification: '1'

---

Based on the following inputs:

Post Title:

{title}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

def thematically_encode_future_withdrawal(state_label, post, title):
    headers = {
        "Content-Type": "application/json"
    }
    if post:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the withdrawl from opioids is in the future tense tense. i.e. planning to withdraw, respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I will begin withdrawal tomorrow"

  - Addiction State Language: "will begin withdrawal" (future tense, referring to withdrawal)

  - Classification: '1'

- Example 2:

  - Post Title: "Starting detox next week"

  - Addiction State Language: "Starting detox next week" (future tense, referring to withdrawal)

  - Classification: '1'

- Example 3:

  - Post Title: "Currently experiencing withdrawal symptoms"

  - Addiction State Language: "experiencing withdrawal symptoms" (present tense, referring to withdrawal)

  - Classification: '0'

- Example 4:

  - Post Title: "Went through withdrawal last month"

  - Addiction State Language: "Went through withdrawal" (past tense, referring to withdrawal)

  - Classification: '0'

- Example 5:

  - Post Title: "Will I survive withdrawal?"

  - Addiction State Language: "Will I survive withdrawal?" (future tense, referring to withdrawal)

  - Classification: '1'

---

Based on the following inputs:

Post Title:

{title}

Post:

{post}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }
    else:
        data = {
            "model": "llama3.2-vision-0temp",
            "messages": [
    {
        "role": "system",
        "content": "You are a researcher in an academic study focused on posts about opiate use on Reddit. Your task is to analyze the addiction state language in the post titles and classify them according to specific rules based on tense and context. Respond with only the specified code '0' or '1'. Do not include reasoning, explanations, or any additional text."
    },
    {
        "role": "user",
        "content": f"""
Instructions:

Analyze the addiction state language in the post title and classify it according to the following rules:

1. Code '1': If the language which refers to the withdrawl from opioids is in the future tense tense. i.e. planning to withdraw, respond with '1'.

2. Code '0': If the above condition is not met, respond with '0'.

Important Notes:

- Definitions of Addiction States Labels:

  - Use: The state of use is characterized by opiate users who are in active use but may have contemplated or made plans for recovery but have not put them into action.

  - Withdrawal: The state of withdrawal is lengthy and often involves difficult withdrawal symptoms. Withdrawal may be assisted through the use of medically prescribed opiates as a substitute, which can last up to 40 days after the start of treatment.

  - Recovery: The state of recovery succeeds a complete withdrawal state and begins when an opiate user's efforts to cease intake have lasted long enough for withdrawal symptoms to largely subside. In this state, recovering users are in stage of maintenance, attempting to remain vigilant of their sobriety throughout the rest of their life.

- Tense refers to the grammatical tense (past, present, future) used when discussing the addiction state.

- State label is an input provided (either 'use', 'withdrawal', or 'recovery').

Examples:

- Example 1:

  - Post Title: "I will begin withdrawal tomorrow"

  - Addiction State Language: "will begin withdrawal" (future tense, referring to withdrawal)

  - Classification: '1'

- Example 2:

  - Post Title: "Starting detox next week"

  - Addiction State Language: "Starting detox next week" (future tense, referring to withdrawal)

  - Classification: '1'

- Example 3:

  - Post Title: "Currently experiencing withdrawal symptoms"

  - Addiction State Language: "experiencing withdrawal symptoms" (present tense, referring to withdrawal)

  - Classification: '0'

- Example 4:

  - Post Title: "Went through withdrawal last month"

  - Addiction State Language: "Went through withdrawal" (past tense, referring to withdrawal)

  - Classification: '0'

- Example 5:

  - Post Title: "Will I survive withdrawal?"

  - Addiction State Language: "Will I survive withdrawal?" (future tense, referring to withdrawal)

  - Classification: '1'

---

Based on the following inputs:

Post Title:

{title}

Post:

{post}

State Label:

{state_label}

---

Your Response:

Respond with digits: '0' or '1'. Do not include any other text."""
    }
],
            "stream": False,
        }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['message']['content']

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
        f.write(f"hallucinations: {num_hallucinations}\n")
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

def encode_tenses(output):
    # present
    if not os.path.exists(f"{output}/present_tense"):
        os.makedirs(f"{output}/present_tense")
    present_csv_path = os.path.join(f"{output}/present_tense", "present_tense_codes.csv")
    with open(present_csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense"])
        writer.writeheader()
        encodings = parse.parse_tense()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            post_id, post, title, state_label, tense = encoding
            try:
                post_id, post, title, state_label, tense = encoding
                # cast tense to binary for present tense
                if 0 in tense:
                    tense = 1
                else:
                    tense = 0
                thematic_code = thematically_encode_present_tense(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_tense": thematic_code,
                    "true_tense": tense
                })
                file.flush()
                true_encodings.append(tense)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(0 if tense == 1 else 1)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
        write_binary_classification_metrics(f"{output}/present_tense", num_hallucinations, true_encodings, predicted_encodings)
    # past_use
    if not os.path.exists(f"{output}/past_use"):
        os.makedirs(f"{output}/past_use")
    past_use_csv_path = os.path.join(f"{output}/past_use", "past_use_codes.csv")
    with open(past_use_csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense"])
        writer.writeheader()
        encodings = parse.parse_tense()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            post_id, post, title, state_label, tense = encoding
            try:
                post_id, post, title, state_label, tense = encoding
                # cast tense to binary
                if 1 in tense:
                    tense = 1
                else:
                    tense = 0
                thematic_code = thematically_encode_past_use(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_tense": thematic_code,
                    "true_tense": tense
                })
                file.flush()
                true_encodings.append(tense)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(0 if tense == 1 else 1)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
        write_binary_classification_metrics(f"{output}/past_use", num_hallucinations, true_encodings, predicted_encodings)
    # past_withdrawal
    if not os.path.exists(f"{output}/past_withdrawal"):
        os.makedirs(f"{output}/past_withdrawal")
    past_withdrawal_csv_path = os.path.join(f"{output}/past_withdrawal", "past_withdrawal_codes.csv")
    with open(past_withdrawal_csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense"])
        writer.writeheader()
        encodings = parse.parse_tense()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            post_id, post, title, state_label, tense = encoding
            try:
                post_id, post, title, state_label, tense = encoding
                # cast tense to binary
                if 2 in tense:
                    tense = 1
                else:
                    tense = 0
                thematic_code = thematically_encode_past_withdrawal(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_tense": thematic_code,
                    "true_tense": tense
                })
                file.flush()
                true_encodings.append(tense)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(0 if tense == 1 else 1)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
        write_binary_classification_metrics(f"{output}/past_withdrawal", num_hallucinations, true_encodings, predicted_encodings)
    # past_recovery
    if not os.path.exists(f"{output}/past_recovery"):
        os.makedirs(f"{output}/past_recovery")
    past_recovery_csv_path = os.path.join(f"{output}/past_recovery", "past_recovery_codes.csv")
    with open(past_recovery_csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense"])
        writer.writeheader()
        encodings = parse.parse_tense()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            post_id, post, title, state_label, tense = encoding
            try:
                post_id, post, title, state_label, tense = encoding
                # cast tense to binary
                if 3 in tense:
                    tense = 1
                else:
                    tense = 0
                thematic_code = thematically_encode_past_recovery(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_tense": thematic_code,
                    "true_tense": tense
                })
                file.flush()
                true_encodings.append(tense)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(0 if tense == 1 else 1)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
        write_binary_classification_metrics(f"{output}/past_recovery", num_hallucinations, true_encodings, predicted_encodings)
    # future withdrawal
    if not os.path.exists(f"{output}/future_withdrawal"):
        os.makedirs(f"{output}/future_withdrawal")
    future_withdrawal_csv_path = os.path.join(f"{output}/future_withdrawal", "future_withdrawal_codes.csv")
    with open(future_withdrawal_csv_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense"])
        writer.writeheader()
        encodings = parse.parse_tense()
        true_encodings = []
        predicted_encodings = []
        num_hallucinations = 0
        for encoding in encodings:
            post_id, post, title, state_label, tense = encoding
            try:
                post_id, post, title, state_label, tense = encoding
                # cast tense to binary
                if 4 in tense:
                    tense = 1
                else:
                    tense = 0
                thematic_code = thematically_encode_future_withdrawal(state_label, post, title)
                writer.writerow({
                    "post_id": post_id,
                    "predicted_tense": thematic_code,
                    "true_tense": tense
                })
                file.flush()
                true_encodings.append(tense)
                try:
                    predicted_encodings.append(int(thematic_code))
                except:
                    num_hallucinations += 1
                    #if the endoding is a hallucination, add a wrong entry to the predictions
                    predicted_encodings.append(0 if tense == 1 else 1)
            except Exception as e:
                print(f"Error processing encoding {encoding}: {e}")
        write_binary_classification_metrics(f"{output}/future_withdrawal", num_hallucinations, true_encodings, predicted_encodings)
    

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


if __name__ == "__main__":
    encode_tenses("llama_thematic_coding/11-21/tenses/run1")