# AKI Predicton Fine-Tuning and Analysis

This repository contains scripts and notebooks for fine-tuning various Large Language Models (LLMs) and clasical machine learning. The repository also contain code for 
comparing energy consumption using text embedding and only tabular feature.

## Data

The dataset used in this project can be downloaded by following the instructions provided in the [Aki-Predictor repository](https://github.com/ExaScience/Aki-Predictor). 
Specifically, we have utilized the "kidigo_7_days_creatinine" dataset. This dataset is derived from a query that checks if the patient had Acute Kidney Injury (AKI) during 
the first 7 days of their ICU stay, according to the KDIGO guidelines and based solely on the creatinine feature.

## Python Scripts

This repository includes six Python scripts for fine-tuning different LLM models. Each script is designed to handle specific aspects of the fine-tuning process, ensuring 
efficient and effective model training.

## Jupyter Notebooks

In addition to the Python scripts, the repository also includes Jupyter notebooks that demonstrate the following:

- **Energy Comparison**: Analyzing the energy consumption of different models.
- **Classical Machine Learning Models**: Training and evaluating classical machine learning models for AKI prediction.

## Usage

To use the resources in this repository:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Follow the instructions in the [Aki-Predictor repository](https://github.com/ExaScience/Aki-Predictor) to download the necessary dataset.

3. Run the desired Python scripts for fine-tuning LLM models:
    ```bash
    python script_name.py
    ```

4. Open and run the Jupyter notebooks to explore energy comparison and classical machine learning models:
    ```bash
    jupyter notebook
    ```

---



