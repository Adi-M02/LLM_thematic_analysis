import json
import parse_codings as parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
import random
import time
import logging
from finetune_encoder import Encoder
import llama_thematic_coding as ltc


val_post_ids = set()
with open("general_finetuning_data_with_post_id/validation_post_ids.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        val_post_ids.add(row[0])

model_category_dict  = {"withdrawal_subs_method_1:latest": ["withdrawal", "subs_method"], "withdrawal_subs_method_3:latest": ["withdrawal", "subs_method"], "withdrawal_subs_method_5:latest": ["withdrawal", "subs_method"], "use_pr_1:latest": ["use", "personal_regimen"], "use_pr_3:latest": ["use", "personal_regimen"], "use_pr_5:latest": ["use", "personal_regimen"]}

def encode_and_evaluate_specific_feature(output, val_post_ids):
    for model, [category, feature] in model_category_dict.items():
        feature_directory = os.path.join(output, category, feature)
        encoder = Encoder(model)
        ltc.create_directory(feature_directory)
        log_file_path = os.path.join(feature_directory, "error_log.txt")
        logger = ltc.setup_logging(log_file_path)
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
                post_id, post, title, state_label, feature_list = encoding
                if post_id not in val_post_ids:
                    continue
                true_label = 1 if ltc.feature_encoding_to_binary(category, feature, feature_list) else 0
                response = encoder.encode(post, title)
                num_errors, predicted_encodings, true_encodings = ltc.write_response_and_update_evaluation_lists(writer, logger, response, post_id, true_label, num_errors, predicted_encodings, true_encodings)
            num_different_examples = ltc.compare_example_and_post(logger, csv_path)
            ltc.write_metrics_and_model(feature_directory, logger, encoder, feature, num_errors, num_different_examples, true_encodings, predicted_encodings)