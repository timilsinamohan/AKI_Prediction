#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
os.environ["WANDB_DISABLED"] = "true"


# Function to set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure PyTorch uses all available threads
torch.set_num_threads(torch.get_num_threads())


# In[ ]:


# df = pd.read_csv("/home/ubuntu/AKI/analysis_aki.csv")
df = pd.read_csv("/home/ubuntu/AKI/analysis_without_urine_aki.csv")

def data_pre_processing(df):
    # Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Automatically detect numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Define transformers for numeric and categorical columns
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create column transformer to apply transformations to the appropriate columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    df_processed = preprocessor.fit_transform(df)
    
    # Convert the processed data back to a DataFrame for better readability
    # Extract feature names for categorical variables
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_cols = numeric_features + list(cat_cols)
    
    # Create the DataFrame
    df_final = pd.DataFrame(df_processed, columns=all_cols)
    return df_final

new_df = data_pre_processing(df)
y = new_df['AKI_7DAY'].values
y = y.astype(int)
new_df = new_df.drop('AKI_7DAY', axis='columns')
new_df.head()



# In[ ]:


sorted(list(new_df.columns))


# In[ ]:


def row_to_sentence_full(row):
    sentence_parts = []
    
    # Starting with patient's age and general description
    if not pd.isna(row['AGE']):
        sentence_parts.append(f"A patient of age {row['AGE']}")

    # Numeric variables (handling a few explicitly for demonstration)
    numeric_cols = [
        'ALBUMIN_MAX', 'ALBUMIN_MIN', 'ANIONGAP_MAX', 'ANIONGAP_MIN', 'BANDS_MAX', 'BANDS_MIN',
        'BICARBONATE_MAX', 'BICARBONATE_MIN', 'BILIRUBIN_MAX', 'BILIRUBIN_MIN', 'BUN_MAX', 'BUN_MIN',
        'CHLORIDE_MAX', 'CHLORIDE_MIN', 'CREAT', 'CREATININE_MAX', 'CREATININE_MIN', 'DIASBP_MAX',
        'DIASBP_MEAN', 'DIASBP_MIN', 'EGFR', 'GLUCOSE_MAX_x', 'GLUCOSE_MAX_y', 'GLUCOSE_MEAN',
        'GLUCOSE_MIN_x', 'GLUCOSE_MIN_y', 'HEARTRATE_MAX', 'HEARTRATE_MEAN', 'HEARTRATE_MIN',
        'HEMATOCRIT_MAX', 'HEMATOCRIT_MIN', 'HEMOGLOBIN_MAX', 'HEMOGLOBIN_MIN', 'INR_MAX', 'INR_MIN',
        'LACTATE_MAX', 'LACTATE_MIN', 'MEANBP_MAX', 'MEANBP_MEAN', 'MEANBP_MIN', 'PLATELET_MAX',
        'PLATELET_MIN', 'POTASSIUM_MAX', 'POTASSIUM_MIN', 'PTT_MAX', 'PTT_MIN', 'PT_MAX', 'PT_MIN',
        'RESPRATE_MAX', 'RESPRATE_MEAN', 'RESPRATE_MIN', 'SODIUM_MAX', 'SODIUM_MIN', 'SPO2_MAX',
        'SPO2_MEAN', 'SPO2_MIN', 'SYSBP_MAX', 'SYSBP_MEAN', 'SYSBP_MIN', 'TEMPC_MAX', 'TEMPC_MEAN',
        'TEMPC_MIN', 'WBC_MAX', 'WBC_MIN'
    ]

    for col in numeric_cols:
        if not pd.isna(row[col]):
            sentence_parts.append(f"{col.replace('_', ' ').lower()} of {row[col]}")

    # Conditions and diseases
    conditions = [
        'ALCOHOL_ABUSE', 'CARDIAC_ARRHYTHMIAS', 'CONGESTIVE_HEART_FAILURE', 'DIABETES_COMPLICATED',
        'DIABETES_UNCOMPLICATED', 'DRUG_ABUSE', 'HYPERTENSION', 'HYPOTHYROIDISM', 'LIVER_DISEASE',
        'OBESITY', 'PERIPHERAL_VASCULAR', 'RENAL_FAILURE', 'VALVULAR_DISEASE'
    ]
    for condition in conditions:
        if row.get(condition, 0) == 1:
            sentence_parts.append(f"has {condition.replace('_', ' ').lower()}")

    # Categorical: Ethnicity and Gender
    ethnicity_cols = [col for col in row.index if "ETHNICITY_" in col]
    for col in ethnicity_cols:
        if row.get(col, 0) == 1:
            sentence_parts.append(f"identified as {col.replace('ETHNICITY_', '').replace('_', ' ').lower()}")

    if row.get('GENDER_F', 0) == 1:
        sentence_parts.append("gender is female")
    elif row.get('GENDER_M', 0) == 1:
        sentence_parts.append("gender is male")

    # Final assembly of the sentence
    if not sentence_parts:
        return "No data available for this patient."
    
    return ', '.join(sentence_parts) + '.'



# In[ ]:


# Example usage with a DataFrame row
row_data = new_df.iloc[0]  # Assuming 'df' is your DataFrame
sentence = row_to_sentence_full(row_data)
print(sentence)


# In[ ]:


texts = new_df.apply(row_to_sentence_full, axis=1).tolist()
labels = y.tolist()

data = {'text': texts,
        'label':labels}
new_df_llm = pd.DataFrame(data)

X = new_df_llm["text"]
y = new_df_llm["label"]

aki_df = new_df_llm[new_df_llm['label'] == 1]
non_aki_df = new_df_llm[new_df_llm['label'] == 0]


# In[ ]:


new_df_llm.head()


# In[ ]:


aki_train, aki_test = train_test_split(new_df_llm, test_size=0.3, random_state=42)
aki_train, aki_val = train_test_split(aki_train, test_size=0.3, random_state=42)

train_dataset = Dataset.from_pandas(aki_train)
test_dataset = Dataset.from_pandas(aki_test)
val_dataset = Dataset.from_pandas(aki_val)


# In[ ]:


# tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess function adjusted for Hugging Face Datasets
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


# In[ ]:


# tokenized_text = data.map(preprocess_function, batched = True)
tokenized_train = train_dataset.map(preprocess_function, batched = True)
tokenized_val = val_dataset.map(preprocess_function, batched = True)
tokenized_test = test_dataset.map(preprocess_function, batched = True)


# In[ ]:


tokenized_train['text'][0]




# Model
model = AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased', num_labels=2).to(device)

model.config.pad_token_id = tokenizer.pad_token_id

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)


# In[ ]:


# Accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)


trainer.train()
logits, _, _ = trainer.predict(tokenized_test)
predictions = np.argmax(logits, axis=-1)
print("accuracy:", accuracy_score(tokenized_test['label'], predictions))
print("precision:", precision_score(tokenized_test['label'], predictions))
print("recall:", recall_score(tokenized_test['label'], predictions))
print("f1:", f1_score(tokenized_test['label'], predictions))
print("auc_roc:", roc_auc_score(tokenized_test['label'], torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()[:, 1]))
print("auc_pr:", average_precision_score(tokenized_test['label'], torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()[:, 1]))


# In[ ]:




