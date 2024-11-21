import pandas as pd
import random
random.seed(0)

df = pd.read_csv("Covid Dataset.csv")
df = df[['Fever', 'Wearing Masks', 'Sore throat', 'Visited Public Exposed Places', 'Running Nose', 'COVID-19']]
print(len(df))

"""
We have a Covid dataset now, but it has over 5000 rows such that human can not look through them all.
In this group activity, you are given a small subset of the original dataset.
Your task is to analyze the small train dataset (not the small test dataset),
and try to get a higher accuracy by building a decision tree with hard-code if-then-else rules.
"""

small_train_dataset = pd.concat([
    df[df['COVID-19'] == 'Yes'].sample(10, random_state=0),
    df[df['COVID-19'] == 'No'].sample(10, random_state=0)
])
small_train_gt = small_train_dataset["COVID-19"]
rest_df = df.drop(index=small_train_dataset.index)

small_test_dataset = pd.concat([
    rest_df[rest_df['COVID-19'] == 'Yes'].sample(10, random_state=0),
    rest_df[rest_df['COVID-19'] == 'No'].sample(10, random_state=0)
])
small_test_gt = small_test_dataset['COVID-19']
small_test_dataset = small_test_dataset.drop(columns=['COVID-19'])
rest_df = rest_df.drop(index=small_test_dataset.index)

large_test_dataset = rest_df.sample(1000, random_state=0)
large_test_gt = large_test_dataset['COVID-19']
large_test_dataset = large_test_dataset.drop(columns=['COVID-19'])
rest_df = rest_df.drop(index=large_test_dataset.index)

# you can see all data in train dataset but not test dataset
print(small_train_dataset)

def model(sample):
    # modify this function to build your own decision tree
    if sample['Running Nose'] == 'Yes':
        return 'Yes'
    else:
        return 'No'

def evaluate(dataset, gt, model):
    right = 0
    for index, sample in dataset.iterrows():
        right += model(sample) == gt[index]
    return right / len(gt)

# run to get accuracy
print("train accuracy:", evaluate(small_train_dataset, small_train_gt, model))
print("small test accuracy:", evaluate(small_test_dataset, small_test_gt, model))
print("large test accuracy:", evaluate(large_test_dataset, large_test_gt, model))
