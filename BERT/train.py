# author: tyler osborne
# 29 November 2022
# BERT training on song lyrics for valence binary classification

# from msilib.schema import Class
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, ClassLabel
# import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# import winsound


# https://huggingface.co/docs/transformers/preprocessing
tokenizer = AutoTokenizer.from_pretrained("trainer_checkpoints\checkpoint-2055", padding="max_length", truncation=True, return_tensor='pt')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# reading in Brendan's lyrics data and preprocessing through HF's auto tokenizer to get it ready for BERT
def preprocess_data():
    df = pd.read_csv('../data_bin.csv')

    # removing songs without lyrics
    toDrop = df.index[df['lyrics'] == 'None'].tolist()
    df.drop(toDrop, axis = 0, inplace = True)

    # making all lyrics max 300 characters
    df['lyrics'] = df['lyrics'].str[:300]

    # print(df['binary_valence'].value_counts())

    # reshaping lyrics to make the sklearn train_test_split function happy
    lyrics_reshaped = np.reshape(df.loc[:, 'lyrics'].to_numpy(), (-1, 1))
    binary_valence_values = np.reshape(df.loc[:, 'binary_valence'].to_numpy(), (-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(lyrics_reshaped, binary_valence_values, test_size=0.2)

    X_train = pd.DataFrame(X_train)
    X_train.rename(columns={0: 'text'}, inplace=True)
    y_train = pd.DataFrame(y_train)
    y_train.rename(columns={0: 'label'}, inplace=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    # print(train_df)

    X_test = pd.DataFrame(X_test)
    X_test.rename(columns={0: 'text'}, inplace=True)
    y_test = pd.DataFrame(y_test)
    y_test.rename(columns={0: 'label'}, inplace=True)
    test_df = pd.concat([X_test, y_test], axis=1)
    # print(test_df)

    train_ds = Dataset.from_pandas(train_df, split='train')
    test_ds = Dataset.from_pandas(test_df, split='test')


    # lyrics
    # X_train = X_train.reshape((-1))
    # X_test = X_test.reshape((-1))

    # return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    return {'train': train_ds, 'test': test_ds}


def get_dataset_dict(data):

    # class_label = ClassLabel(num_classes=2, names=['High valence', 'Low valence'], id=None)

    train_dataset = Dataset.from_dict(
        {"text": data['train']['X'].tolist(), "label": data['train']['y'].tolist()}
    )
    test_dataset = Dataset.from_dict(
        {"text": data['test']['X'].tolist(), "label": data['test']['y'].tolist()}
    )
    return DatasetDict({'train': train_dataset, 'eval': test_dataset})

# computing evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc.tolist(),
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }

data = preprocess_data()
train_tokenized_data = data['train'].map(tokenize_function, batched=True)
test_tokenized_data = data['test'].map(tokenize_function, batched=True)


# initializing the model with 2 labels, one for positive valence and one for negative
def model_init():
    return AutoModelForSequenceClassification.from_pretrained("trainer_checkpoints\checkpoint-2055", num_labels=2)

# model = model_init()

# setting up training checkpoints
training_args = TrainingArguments(output_dir="trainer_checkpoints", evaluation_strategy='epoch', learning_rate=2e-5, num_train_epochs=5, \
    load_best_model_at_end=False, save_strategy='epoch', per_device_train_batch_size=8, per_device_eval_batch_size=8)


trainer = Trainer(
    # model_init=model_init,
    model=model_init(),
    args=training_args,
    train_dataset=train_tokenized_data,
    eval_dataset=test_tokenized_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# trainer.train()
# results = str(trainer.evaluate())
# f = open('results.txt', 'w')
# f.write(results)
# f.close()
# print(results)

# best_run = trainer.hyperparameter_search(n_trials=10, direction='maximize')
# print(best_run)
trainer.train()
print(str(trainer.evaluate()))
# winsound.Beep(440, 3000)

