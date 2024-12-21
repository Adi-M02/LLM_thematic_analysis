import csv
import llama_thematic_coding as ltc
import os
from pathlib import Path
import logging

def metrics_on_val_posts(post_id_file, codes_file, output):
    post_ids = set()
    with open(post_id_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            post_ids.add(row[0])

    with open(codes_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        post_id_index = header.index("post_id")
        predicted_index = header.index("predicted_tense")
        true_index = header.index("true_tense")
        post_id_code = {}
        predicted_list = []
        true_list = []
        num_errors = 0
        for row in reader:
            try:
                post_id = row[post_id_index]
                if post_id in post_ids:
                    try:
                        true = int(row[true_index])
                        predicted = int(row[predicted_index])
                        predicted_list.append(true)
                        true_list.append(predicted)
                    except:
                        num_errors += 1
            except:
                num_errors += 1
        ltc.write_metrics_and_model(output, "", "", "", num_errors, 0, true_list, predicted_list)

def all_metrics_on_val_posts(input_dir, output_dir, val_post_ids_file):
    results = {}
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        logging.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
        return results

    for file_path in input_path.rglob("*_codes*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            logging.info(f"Processing file: {relative_path}")
            
            # Extract directory components excluding the file name
            dir_parts = relative_path.parent.parts  # Tuple of directories
            file_stem = file_path.stem  # Filename without extension

            # Construct the corresponding output directory path
            output_subdir = output_path.joinpath(*dir_parts)
            output_subdir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured directory exists: {output_subdir}")

            # Construct the output file path with .csv extension
            output_file = output_subdir / f"{file_stem}.csv"

            # Call the metrics function
            print(f"Processing {file_path}")
            metrics = metrics_on_val_posts(val_post_ids_file, str(file_path), str(output_file))
            results[str(relative_path)] = metrics

    return results

if __name__ == "__main__":
    val_post_ids_file = "general_finetuning_data_with_post_id/validation_post_ids.csv"
    all_metrics_on_val_posts("llama_thematic_coding/12-7/test4", "val_set_results", val_post_ids_file)
    