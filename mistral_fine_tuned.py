#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import random
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
import evaluate
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,precision_recall_fscore_support
from sklearn.metrics import (accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            roc_auc_score,
                            average_precision_score,
                            precision_recall_fscore_support,
                            precision_recall_curve,
                            auc)
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import KNNImputer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from trl import SFTTrainer
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import functools
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
ds = DatasetDict({'train': train_dataset, 'val': val_dataset,'test':test_dataset})

# model name
model_name = 'mistralai/Mistral-7B-v0.1'

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['label']
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
tokenized_ds = tokenized_ds.with_format('torch')

# qunatization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

# lora config
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)
# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=2
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits[:, 0] > 0).astype(int)
    
    # Calculate basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    # Calculate probabilities for the positive class
    probabilities = torch.sigmoid(torch.tensor(logits[:, 0])).numpy()
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(labels, probabilities)
    
    # Calculate AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
    auc_pr = auc(recall_curve, precision_curve)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }

class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss()
        # No need to apply sigmoid or softmax as CrossEntropyLoss does that internally
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# define training args
training_args = TrainingArguments(
    output_dir = 'binary_classification',
    learning_rate = 1e-4,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 3,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True
)

# train
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_ds['train'],
    eval_dataset = tokenized_ds['val'],
    tokenizer = tokenizer,
    data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics = compute_metrics
  
)


trainer.train()
logits, _, _ = trainer.predict(tokenized_ds['test'])
predictions = np.argmax(logits, axis=-1)
print("accuracy:", np.around(accuracy_score(tokenized_ds['test']['labels'], predictions), decimals = 3))
print("precision:", np.around(precision_score(tokenized_ds['test']['labels'], predictions),decimals=3))
print("recall:", np.around(recall_score(tokenized_ds['test']['labels'], predictions),decimals=3))
print("f1:", np.around(f1_score(tokenized_ds['test']['labels'], predictions),decimals=3))
print("auc_roc:", np.around(roc_auc_score(tokenized_ds['test']['labels'], torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()[:, 1]),decimals=3))
print("auc_pr:", np.around(average_precision_score(tokenized_ds['test']['labels'], torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()[:, 1]),decimals=3))



