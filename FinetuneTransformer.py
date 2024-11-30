import pandas as pd
from sentence_transformers import (
    losses,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
# from datasets import load_dataset
from datasets import Dataset, DatasetDict

df = pd.read_csv("Dataset/labeled_data.csv", usecols=['label','body_parent','body_child','individual_kappa'])
#CoSENTLoss
# df = df.dropna()
# df.loc[df['label'] == 1, 'label'] = 0.5
# df.loc[df['label'] == 2, 'label'] = 0.5 + (0.5*df.loc[df['label'] == 2, 'individual_kappa'])
# df.loc[df['label'] == 0, 'label'] = 0.5 - (0.5*df.loc[df['label'] == 0, 'individual_kappa'])
# df = df.drop(['individual_kappa'], axis=1)
# df = df.iloc[:, [1, 2, 0]]
#SoftmaxLoss
def SoftMax_arrange(df):
    temp_df = temp_df.drop(['individual_kappa'], axis=1)
    temp_df = temp_df.dropna()
    temp_df = temp_df.iloc[:, [1, 2, 0]]
#ContrastLoss
#You must use BatchSamplers.GROUP_BY_LABEL
def ContrastLoss_arrange(agree, df, batchsize):
    temp_df = df.copy(deep=True)
    temp_df = temp_df.drop(['individual_kappa'], axis=1)
    temp_df = temp_df.dropna()
    # temp_df['combined'] = temp_df['body_parent'] + " !|! " + temp_df['body_child'] 
    # temp_df = temp_df.drop(['body_parent', 'body_child', 'individual_kappa'], axis=1)

    # temp_df.loc[temp_df['label'] != 2, 'label'] = 0
    # temp_df.loc[temp_df['label'] == 2, 'label'] = 1
    temp_df.loc[temp_df['label'] != 0, 'label'] = -1
    temp_df.loc[temp_df['label'] == 0, 'label'] = 1
    temp_df.loc[temp_df['label'] == -1, 'label'] = 0

    # temp_df = temp_df.iloc[:, [1, 0]]
    temp_df = temp_df.iloc[:, [1, 2, 0]]
    # num_rows = len(temp_df)
    # half_size = num_rows // 2  # Integer division, removes remainder if odd

    # Step 2: Split the DataFrame into two halves
    # df1 = temp_df.iloc[:half_size].reset_index(drop=True)
    # df2 = temp_df.iloc[half_size:2*half_size].reset_index(drop=True)
    # combined = temp_df.columns.get_loc('combined')
    # label = temp_df.columns.get_loc('label')

    # new_df = pd.DataFrame({
    #     'combined_1': [df1.iloc[i, combined] for i in range(half_size)],
    #     'combined_2': [df2.iloc[i, combined] for i in range(half_size)],
    #     'label': [1 if (df1.iloc[i, label] == df2.iloc[i, label]) else 0 for i in range(half_size)],
    # })
    return temp_df

    # print(df.to_string())
    # exit()

df = ContrastLoss_arrange(True, df, 32)
size = len(df)
# print(df)
# print((df['label'] == 1).sum())
# print((df['label'] == 0).sum())
# exit()

train_df = df.head(size-2000)
train_data = Dataset.from_pandas(train_df, preserve_index=False)

eval_df = df.iloc[size-2000:size-1000]
eval_data = Dataset.from_pandas(eval_df, preserve_index=False)

#model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Prepare loss function
# loss = losses.CoSENTLoss(model)
# loss = losses.SoftmaxLoss(model, model.get_sentence_embedding_dimension(), num_labels=3)
loss = losses.OnlineContrastiveLoss(model)

# args
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="transformers/disagree_ft_mpnet_Constrast",
    learning_rate=2e-5,
    weight_decay=3e-3,
    num_train_epochs=3,
    load_best_model_at_end=True,
    # batch_sampler=BatchSamplers.GROUP_BY_LABEL,
    #validation stuff
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    logging_steps=100,
)

#validation model
dev_evaluator = BinaryClassificationEvaluator(
    sentences1=eval_data["body_parent"],
    sentences2=eval_data["body_child"],
    labels=eval_data["label"],
    name="contrast_dev",
)
dev_evaluator(model)

# Initialize Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()