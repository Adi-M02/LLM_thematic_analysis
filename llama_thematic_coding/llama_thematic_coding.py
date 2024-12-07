import requests
import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
import random
import time
import logging
from thematic_encoder import ThematicEncoder

category_feature_dict = {
  "tense": [
      "present_tense",
      "past_use",
      "past_withdrawal",
      "past_recovery",
      "future_withdrawal"
  ],
  "atypical_information": [
      "want_to_use",
      "talking_about_withdrawal",
      "talking_about_use",
      "mentioning_withdrawal_drugs",
      "not_mentioning_withdrawal"
  ],
  "special_cases": [
      "relapse_mention",
      "unintentional_withdrawal",
      "abusing_subs",
      "irregular_use",
      "use_for_pain_relief"
  ],
  "use": [
      "personal_regimen",
      "improper_administration",
      "purchase_of_drugs",
      "negative_effects",
      "activity_on_opiates",
      "positive_effects"
  ],
  "withdrawal": [
      "subs_method",
      "methadone_method",
      "zolpiclone_method",
      "diazepam_method",
      "kratom_method",
      "unmentioned_method",
      "xanax_method",
      "sleeping_pills_method",
      "loperamide_method",
      "marijuana_method",
      "gabapentin_method",
      "klonopin_method",
      "rhodiola_method",
      "vivitrol_method",
      "cigarette_methods",
      "caffine_method",
      "cold_turkey_method",
      "ibogaine_method",
      "restless_legs_symptom",
      "sleep_disorder_symptom",
      "GI_symptom",
      "sweats_symptom",
      "cold_sensitivity_symptom",
      "nausea_vomiting_symptom",
      "memory_loss_symptom",
      "heartburn_symptom",
      "headache_symptom",
      "sore_throat_symptom",
      "cold_flu_fever_symptom"
  ],
  "recovery": [
      "offering_advice",
      "challenges_through_recovery",
      "danger_of_opiates"
  ],
  "co-use": [
      "xanax",
      "benzodiazepam",
      "ambien",
      "aderall",
      "marijuana",
      "cigarettes",
      "cocaine",
      "ketorolac",
      "vinegar",
      "alcohol",
      "amphetamine",
      "imodium"
  ],
  "off-topic": [
      "public_health_awareness",
      "seeking_community",
      "other_persons_opiate_use", 
      "entertainment"
  ],
  "question": [
      "opioid_use_lifestyle",
      "technical_drug_use",
      "effects",
      "methadone",
      "suboxone",
      "improper_use",
      "misc_question",
      "subutex",
      "tramadol",
      "weed",
      "kratom",
      "darvocet",
      "vivitrol",
      "relate_to_defeated",
      "relate_to_recovery",
      "relate_to_withdrawal",
      "relate_to_using",
      "deal_with_relapse",
      "recover_again",
      "resetting_withrawal",
      "withdrawal",
      "withdrawal_symptoms",
      "effects_of_withdrawal",
      "withdrawal_pain",
      "recovery_question",
      "life_without_drugs",
      "non-opiate_medication_question",
      "NA_meeting_question"
  ]
}
feature_prompt_dict = {
    "present_tense": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

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
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "past_use": """Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to the use of opiates is in the past tense, and the state label is 'withdrawal' or 'recovery'.
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
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "past_withdrawal": """Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to the withdrawal from opiates is in the past tense, and the state label is 'use' or 'recovery'.
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
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "past_recovery": """Consider the addiction state label and the addiction state language in the post and post title and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to recovery from opiates is in the past tense, and the state label is 'use' or 'withdrawal'.
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
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "future_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if all the language which refers to withdrawal from opiates is in the future tense.
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
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "want_to_use": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if the addiction state label is 'withdrawal' and the user expresses a desire to use opiates.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the addiction state label is 'use' or 'recovery' or if the user does not express desire to use opiates. 
    - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "talking_about_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if addiction state label is 'use' or 'recovery' and the user is talking about withdrawal symptoms, methods, etc.
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the addiction state label is 'withdrawal' or if the user does not talk about withdrawal symptoms, methods, etc. 
    - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "talking_about_use": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if addiction state label is 'withdrawal' or 'recovery' and the user refers to the act of using opiates. 
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the addiction state label is 'use' or if the user does not talk about the act of using opiates. 
    - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "mentioning_withdrawal_drugs": """ Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
    - Assign label '1' if addiction state label is 'use' or 'recovery' and the user mentions opiate withdrawal drugs such as suboxone, kratom, methadone, etc. 
    - Provide a verbatim section of the text that supports the label.

  2. Label '0':
    - Assign label '0' if the addiction state label is 'withdrawal' or if the user does not mention opiate withdrawal drugs such as suboxone, kratom, methadone, etc. 
    - Respond 'None' in the section of your response that supports the label.

- Important Notes:
  - Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
  - Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
  - Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
  - Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "not_mentioning_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if addiction state label is 'withdrawal' and the user does not mention their personal opiate withdrawal symptoms. 
  - Respond 'None' in the section of your response that supports the label.

2. Label '0':
  - Assign label '0' if the addiction state label is 'use' or 'recovery' or if the user mentions their personal opiate withdrawal symptoms.
  - Provide a verbatim section of the text that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "relapse_mention": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user mentions having relapsed frm opiate recovery or opiate withdrawal.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention having relapsed frm opiate recovery or opiate withdrawal.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "unintentional_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user is in withdrawal from opiates because they are unable to acquire more opiates.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is able to acquire opiates or does not mention being in opiate withdrawal. 
  - Provide a verbatim section of the text that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "abusing_subs": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user is using withdrawal drugs such as suboxone, kratom, methadone, etc. recreationally and does not intend to cease or lower their intake. 
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is not using withdrawal drugs such as suboxone, kratom, methadone, etc. recreationally or is using these but intends to cease or lower their intake.
  - Provide a verbatim section of the text that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "irregular_use": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user is engaged in irregular opiate use, which is periods of abstinence from opiates with the intention of returning to using opiates, or maintaining a reduced opiate intake without the intention to cease use, or withdrawing from opiates for the purpose of lowering their tolerance. 
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is not engaged in irregular opiate use.
  - Provide a verbatim section of the text that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "use_for_pain_relief": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user mentions using opiates for pain relief. 
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using opiates for pain relief. 
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "personal_regimen": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user refers to their personal opiate regimen. 
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not refer to their personal opiate regimen. 
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "improper_administration": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if user refers to improper administration of opiates. Which is snorting, shooting, or anything other than taking a pill to administer opiates. 
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not refer to improper administration of opiates.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "purchase_of_drugs": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' the user discusses anything related to the purchase of drugs.
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not discuss anything related to the purchase of drugs. 
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "negative_effects": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user refers to the effects of opiate use with a negative connotation.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not refer to the effects of opiate use with a negative connotation. 
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "activity_on_opiates": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user refers to an activity they engage in while on opiates.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not refer to an activity they engage in while on opiates.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "positive_effects": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user refers to the effects of opiates use with a positive connotation.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not refer to the effects of opiate use with a positive connotation. 
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "subs_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving subutex/suboxone to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving subutex/suboxone to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "methadone_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving methadone to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving methadone to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "zolpiclone_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving zolpiclone to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving zolpiclone to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "diazepam_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving diazepam to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving diazepam to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "kratom_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving kratom to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving kratom to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "unmentioned_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method other than subutex/suboxon, methadone, zolpiclone, diazepam, kratom, xanax, sleeping pills, loperamide, marijuana, gabapentin, klonopin, rhodiola, vivitrol, cigarettes, cafeeine, going cold turkey, or ibogaine to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "xanax_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving xanax to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving xanax to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "sleeping_pills_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving sleeping pills to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving sleeping pills to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "loperamide_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving loperamide to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving loperamide to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "marijuana_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving marijuana to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving marijuana to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "gabapentin_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving gabapentin to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving gabapentin to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "klonopin_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving klonopin to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving klonopin to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "rhodiola_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving rhodiola to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving rhodiola to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "vivitrol_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving vivitrol to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving vivitrol to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "cigarette_methods": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving cigarettes to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving cigarettes to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "caffine_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving caffeine to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving caffeine to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "cold_turkey_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving going cold turkey to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving going cold turkey to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "ibogaine_method": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using a method involving ibogaine to aid in opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not mention using a method involving ibogaine to aid in opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "restless_legs_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes restless legs as a symtpom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe restless legs as a symtpom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "sleep_disorder_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes being sleepless as a symtpom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe being sleepless as a symtpom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "GI_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes gastrointestinal symptoms ,such as heartburn, indigestion, bloating, or constipation, as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe gastrointestinal symptoms as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "sweats_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes sweats as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe sweats as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "cold_sensitivity_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes cold sensitivity as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe cold sensitivity as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "nausea_vomiting_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes nausea and or vomiting as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe nausea and or vomiting as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "memory_loss_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes memory loss as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe memory loss as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "heartburn_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes heartburn as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe heartburn as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "headache_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes headaches as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe headaches as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "sore_throat_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes sore throats as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe sore throats as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "cold_flu_fever_symptom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes the cold, flu or fevers as a symptom of their opiate withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe the cold, flu or fevers as a symptom of their opiate withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "offering_advice": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user offers advice on opiate recovery or withdrawal.  
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not offer advice on opiate recovery or withdrawal.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "challenges_through_recovery": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes challenges and accomplishments throughout the process of opiate withdrawal and recovery.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe challenges and accomplishments throughout the process of opiate withdrawal and recovery.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "danger_of_opiates": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes the dangers of using opiates.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe the dangers of using opiates.   
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "xanax": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and xanax.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and xanax.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "benzodiazepam": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and benzodiazepam.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and benzodiazepam.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "ambien": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and ambien.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and ambien.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "aderall": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and aderall.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and aderall.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "marijuana": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and marijuana.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and marijuana.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "cigarettes": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and cigarettes.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and cigarettes.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "cocaine": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and cocaine.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and cocaine.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "ketorolac": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and ketorolac.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and ketorolac.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "vinegar": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and vinegar.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and vinegar.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "alcohol": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and alcohol.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and alcohol.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "amphetamine": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and amphetamines.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and amphetamines.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "imodium": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user describes using both opiates and imodium.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not describe using both opiates and imodium.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "public_health_awareness": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user is making a public health awareness statement regarding opiates.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is not making a public health awareness statement regarding opiates.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "seeking_community": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user is seeking community with other users in the post.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is not seeking community with other users in their post.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "other_persons_opiate_use": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user is referring to another persons opiate use.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is not referring to another persons opiate use.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "entertainment": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user makes a statement regarding entertainment or news with a link to an article or video.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user is does not make a statement regarding entertainment or the news with no link to an article or video.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "opioid_use_lifestyle": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the opiate use lifestyle. That is any question about opiates that is not spicifically about administration or effects.   
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the opiate use lifestyle.  
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "technical_drug_use": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about technical opiate use. That is any question about opiates specifically about administration or effects.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about technical opiate use.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "effects": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about effects of opiates.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the effects of opiates.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "methadone": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug methadone.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug methadone.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "suboxone": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug suboxone.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug suboxone.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "improper_use": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about improper use for withdrawal.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about improper use for wihdrawal.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "misc_question": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a general question unrelated to opiate use.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "subutex": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug subutex.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug subutex.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "tramadol": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug tramadol.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug tramadol.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "weed": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug weed.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug weed.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "kratom": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug kratom.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug kratom.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "darvocet": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug darvocet.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug darvocet.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "vivitrol": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

  1. Label '1':
  - Assign label '1' if the user asks a question about the withdrawal drug vivitrol.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about the withdrawal drug vivitrol.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.
  """,
  "relate_to_defeated": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks if others can relate to the experience of being defeated.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask if others can relate to the experience of being defeated.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "relate_to_recovery": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks if others can relate to the experience of recovery.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask if others can relate to the experience of recovery.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "relate_to_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks if others can relate to the experience of withdrawal.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask if others can relate to the experience of withdrawal.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "relate_to_using": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks if others can relate to the experience of using.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask if others can relate to the experience of using.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "deal_with_relapse": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks how to deal with a relapse.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask how to deal with a relapse.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "recover_again": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks how to deal with a recover again after a relapse.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask how to recover again after a relapse.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "resetting_withrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks about resetting withdrawal time after a relapse.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask about resetting withdrawal time after a relapse.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks general questions about withdrawal.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask general questions about withdrawal.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "withdrawal_symptoms": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks how bad withdrawal symptoms will be.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask how bad withdrawal symptoms will be.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "effects_of_withdrawal": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks about effects of withdrawal.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask about effects of withdrawal.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "withdrawal_pain": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks about how to deal with pain without opiates.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask about how to deal with pain without opiates.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "recovery_question": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks about recovery, the period of time after withdrawal symptoms have passed.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask about recovery.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "life_without_drugs": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks about how to have a life in recovery or withdrawal without drugs.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask about how to have a life in recovery or withdrawal without drugs.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term.""",
  "non-opiate_medication_question": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a non-opiate medication question.    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a non-opiate medication question.
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term""",
  "NA_meeting_question": """Analyze the addiction state language in the post and post title, and classify it according to the following rules:

1. Label '1':
  - Assign label '1' if the user asks a question about NA (Narcotics Anonymous).    
  - Provide a verbatim section of the text that supports the label.

2. Label '0':
  - Assign label '0' if the user does not ask a question about NA (Narcotics Anonymous).
  - Respond 'None' in the section of your response that supports the label.

- Important Notes:
- Addiction state language refers to mentions of use, withdrawal, or recovery related to opiate addiction.

- Definitions of Addiction States:
- Use: The user is engaged in opiate use without consideration of quitting or expressing desire to stop using opiates to prepare to quit. 
- Withdrawal: The user has ceased or lowered their opiate intake. Opiate withdrawal is accompanied by a combination of physical and emotional symptoms.
- Recovery: The user has finished detoxing and is attempting to sustain abstinence from opiates long term"""
}
  
def write_binary_classification_metrics(output_dir, num_hallucinations, num_different_examples, true_encodings, predicted_encodings):
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

def write_metrics_and_model(output_dir, logger, encoder, feature, num_hallucinations, num_different_examples, true_encodings, predicted_encodings):
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
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='macro', zero_division=0):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='macro', zero_division=0):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='macro', zero_division=0):.4f}\n")
            f.write(f"- Weighted Averages:\n")
            f.write(f"    - F1 Score: {f1_score(true_encodings, predicted_encodings, average='weighted', zero_division=0):.4f}\n")
            f.write(f"    - Precision: {precision_score(true_encodings, predicted_encodings, average='weighted', zero_division=0):.4f}\n")
            f.write(f"    - Recall: {recall_score(true_encodings, predicted_encodings, average='weighted', zero_division=0):.4f}\n")
            # Create Confusion Matrix
            cm = confusion_matrix(true_encodings, predicted_encodings)
            f.write("Confusion Matrix:\n")
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            f.write(f"    [[TP: {tp} ({(tp / total) * 100:.2f}%), FP: {fp} ({(fp / total) * 100:.2f}%)]\n")
            f.write(f"     [FN: {fn} ({(fn / total) * 100:.2f}%), TN: {tn} ({(tn / total) * 100:.2f}%)]]\n")
        except Exception as e:
            f.write(f"Error calculating metrics: {e}\n")
            logger.error(f"{feature}: Error calculating metrics: {e}")
        encoder.write_prompt_structure(f, feature_prompt_dict[feature])

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
    # Get a unique logger name based on the log file path
    logger_name = tense_log_identifier(log_file_path)
    logger = logging.getLogger(logger_name)
    
    # If the logger is already set up, return it
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)

    # Set up file handler with the given log file path
    handler = logging.FileHandler(log_file_path, mode='a')  # Use 'a' to append logs
    handler.setLevel(logging.INFO)

    # Set log format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

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

def feature_encoding_to_binary(category, feature, encoded_label_list):
    if category == "tense":
        if feature == "present_tense":
            return 0 in encoded_label_list
        elif feature == "past_use":
            return 1 in encoded_label_list
        elif feature == "past_withdrawal":
            return 2 in encoded_label_list
        elif feature == "past_recovery":
            return 3 in encoded_label_list
        elif feature == "future_withdrawal":
            return 4 in encoded_label_list
    elif category == "atypical_information":
        if feature == "want_to_use":
            return 1 in encoded_label_list
        elif feature == "talking_about_withdrawal":
            return 2 in encoded_label_list
        elif feature == "talking_about_use":
            return 3 in encoded_label_list
        elif feature == "mentioning_withdrawal_drugs":
            return 4 in encoded_label_list
        elif feature == "not_mentioning_withdrawal":
            return 5 in encoded_label_list
    elif category == "special_cases":
        if feature == "relapse_mention":
            return 1 in encoded_label_list
        elif feature == "unintentional_withdrawal":
            return 2 in encoded_label_list
        elif feature == "abusing_subs":
            return 3 in encoded_label_list
        elif feature == "irregular_use":
            return 4 in encoded_label_list
        elif feature == "use_for_pain_relief":
            return 5 in encoded_label_list
    elif category == "use":
        if feature == "personal_regimen":
            return 1 in encoded_label_list
        elif feature == "improper_administration":
            return 2 in encoded_label_list
        elif feature == "purchase_of_drugs":
            return 4 in encoded_label_list
        elif feature == "negative_effects":
            return 5 in encoded_label_list
        elif feature == "activity_on_opiates":
            return 7 in encoded_label_list
        elif feature == "positive_effects":
            return 8 in encoded_label_list
    elif category == "withdrawal":
        if feature == "subs_method":
            return 1 in encoded_label_list
        elif feature == "methadone_method":
            return 2 in encoded_label_list
        elif feature == "zolpiclone_method":
            return 3 in encoded_label_list
        elif feature == "diazepam_method":
            return 4 in encoded_label_list
        elif feature == "kratom_method":
            return 5 in encoded_label_list
        elif feature == "unmentioned_method":
            return 6 in encoded_label_list
        elif feature == "xanax_method":
            return 7 in encoded_label_list
        elif feature == "sleeping_pills_method":
            return 8 in encoded_label_list
        elif feature == "loperamide_method":
            return 9 in encoded_label_list
        elif feature == "marijuana_method":
            return 12 in encoded_label_list
        elif feature == "gabapentin_method":
            return 13 in encoded_label_list
        elif feature == "klonopin_method":
            return 14 in encoded_label_list
        elif feature == "rhodiola_method":
            return 15 in encoded_label_list
        elif feature == "vivitrol_method":
            return 26 in encoded_label_list
        elif feature == "cigarette_methods":
            return 27 in encoded_label_list
        elif feature == "caffine_method":
            return 28 in encoded_label_list
        elif feature == "cold_turkey_method":
            return 29 in encoded_label_list
        elif feature == "ibogaine_method":
            return 20 in encoded_label_list
        elif feature == "restless_legs_symptom":
            return 10 in encoded_label_list
        elif feature == "sleep_disorder_symptom":
            return 11 in encoded_label_list
        elif feature == "GI_symptom":
            return 16 in encoded_label_list
        elif feature == "sweats_symptom":
            return 17 in encoded_label_list
        elif feature == "cold_sensitivity_symptom":
            return 18 in encoded_label_list
        elif feature == "nausea_vomiting_symptom":
            return 19 in encoded_label_list
        elif feature == "memory_loss_symptom":
            return 21 in encoded_label_list
        elif feature == "heartburn_symptom":
            return 22 in encoded_label_list
        elif feature == "headache_symptom":
            return 23 in encoded_label_list
        elif feature == "sore_throat_symptom":
            return 24 in encoded_label_list
        elif feature == "cold_flu_fever_symptom":
            return 25 in encoded_label_list
    elif category == "recovery":
        if feature == "offering_advice":
            return 1 in encoded_label_list
        elif feature == "challenges_through_recovery":
            return 2 in encoded_label_list
        elif feature == "danger_of_opiates":
            return 3 in encoded_label_list
    elif category == "co-use":
        if feature == "xanax":
            return 1 in encoded_label_list
        elif feature == "benzodiazepam":
            return 2 in encoded_label_list
        elif feature == "ambien":
            return 3 in encoded_label_list
        elif feature == "aderall":
            return 4 in encoded_label_list
        elif feature == "marijuana":
            return 5 in encoded_label_list
        elif feature == "cigarettes":
            return 6 in encoded_label_list
        elif feature == "cocaine":
            return 7 in encoded_label_list
        elif feature == "ketorolac":
            return 8 in encoded_label_list
        elif feature == "vinegar":
            return 9 in encoded_label_list
        elif feature == "alcohol":
            return 10 in encoded_label_list
        elif feature == "amphetamine":
            return 11 in encoded_label_list
        elif feature == "imodium":
            return 12 in encoded_label_list
    elif category == "off-topic":
        if feature == "public_health_awareness":
            return 1 in encoded_label_list
        elif feature == "seeking_community":
            return 5 in encoded_label_list
        elif feature == "other_persons_opiate_use":
            return 7 in encoded_label_list
        elif feature == "entertainment":
            return 8 in encoded_label_list
    elif category == "question":
        if feature == "opioid_use_lifestyle":
            return 1 in encoded_label_list
        elif feature == "technical_drug_use":
            return 2 in encoded_label_list
        elif feature == "effects":
            return 3 in encoded_label_list
        elif feature == "methadone":
            return 4 in encoded_label_list
        elif feature == "suboxone":
            return 5 in encoded_label_list
        elif feature == "improper_use":
            return 6 in encoded_label_list
        elif feature == "misc_question":
            return 7 in encoded_label_list
        elif feature == "subutex":
            return 32 in encoded_label_list
        elif feature == "tramadol":
            return 8 in encoded_label_list
        elif feature == "weed":
            return 25 in encoded_label_list
        elif feature == "kratom":
            return 26 in encoded_label_list
        elif feature == "darvocet":
            return 28 in encoded_label_list
        elif feature == "vivitrol":
            return 29 in encoded_label_list
        elif feature == "relate_to_defeated":
            return 11 in encoded_label_list
        elif feature == "relate_to_recovery":
            return 12 in encoded_label_list
        elif feature == "relate_to_withdrawal":
            return 20 in encoded_label_list
        elif feature == "relate_to_using":
            return 27 in encoded_label_list
        elif feature == "deal_with_relapse":
            return 29 in encoded_label_list
        elif feature == "recover_again":
            return 33 in encoded_label_list
        elif feature == "resetting_withrawal":
            return 18 in encoded_label_list
        elif feature == "withdrawal":
            return 13 in encoded_label_list
        elif feature == "withdrawal_symptoms":
            return 14 in encoded_label_list
        elif feature == "effects_of_withdrawal":
            return 15 in encoded_label_list
        elif feature == "withdrawal_pain":
            return 16 in encoded_label_list
        elif feature == "recovery_question":
            return 23 in encoded_label_list
        elif feature == "life_without_drugs":
            return 24 in encoded_label_list
        elif feature == "non-opiate_medication_question":
            return 17 in encoded_label_list
        elif feature == "NA_meeting_question":
            return 21 in encoded_label_list

def compare_example_and_post(logger, llm_output):
    modified_llm_output = []
    num_diff = 0
    with open(llm_output, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            post = parse.get_post_title_string(logger, row['post_id'])
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

def write_response_and_update_evaluation_lists(writer, logger, response, post_id, true_tense, num_errors, predicted_encodings, true_encodings):
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
            "true_tense": true_tense, 
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
        return num_errors, predicted_encodings, true_encodings
    try: 
        code = int(thematic_code)
        if code not in [0, 1]:
            num_errors += 1
            logger.error(f"Error appending: post id: {post_id}, {thematic_code}")
            return num_errors, predicted_encodings, true_encodings
        predicted_encodings.append(code)
        true_encodings.append(true_tense)
        return num_errors, predicted_encodings, true_encodings
    except:
        num_errors += 1
        logger.error(f"Error appending: post id: {post_id}, {thematic_code}")
        return num_errors, predicted_encodings, true_encodings

def encode_features(output, category_feature_dict = category_feature_dict):
    for category in category_feature_dict:
        directory_path = os.path.join(output, category)
        encoder = ThematicEncoder()
        create_directory(directory_path)
        log_file_path = os.path.join(directory_path, "error_log.txt")
        logger = setup_logging(log_file_path)
        for feature in category_feature_dict[category]:
            feature_directory = os.path.join(directory_path, feature)
            create_directory(feature_directory)
            csv_path = os.path.join(feature_directory, f"{feature}_codes.csv")
            with open(csv_path, 'w', newline='', encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["post_id", "predicted_tense", "true_tense", "verbatim_example", "exact_match"])
                writer.writeheader()
                encodings = parse.parse_feature(category)
                true_encodings = []
                predicted_encodings = []
                num_errors = 0
                for encoding in encodings:
                    file.flush()
                    post_id, post, title, state_label, tense_list = encoding
                    true_tense = 1 if feature_encoding_to_binary(category, feature, tense_list) else 0
                    response = encoder.encode(feature_prompt_dict[feature], post, title, state_label)
                    num_errors, predicted_encodings, true_encodings = write_response_and_update_evaluation_lists(writer, logger, response, post_id, true_tense, num_errors, predicted_encodings, true_encodings)
                num_different_examples = compare_example_and_post(logger, csv_path)
                write_metrics_and_model(feature_directory, logger, encoder, feature, num_errors, num_different_examples, true_encodings, predicted_encodings)
                    




            
if __name__ == "__main__":
    start = time.time()
    encode_features("llama_thematic_coding/12-7/test4")
    print(f"Time taken: {((time.time() - start)/60):.2f} minutes")
