LLaMA Thematic Coding Project
Setup and Usage

Activate the conda environment using the provided configuration:

Copyconda env create -f condaenv.yaml
conda activate <environment_name>

Run thematic coding on posts:

python llama_thematic_coding/llama_thematic_coding.py

Finetune the model on training data:

python finetuning/finetune_with_unsloth.py
Training sets can be found in the finetuning_data/ directory.
Results

Thematic coding results are located in llama_thematic_coding/12-7/
Summary metrics can be found in:

12-7_test4_metrics_summary.xlsx
val_set_results_metrics.xlsx