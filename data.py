import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

def add_labels(data):
    """adds labels to corresponding clips"""
    labels_file_path = "data/observationsALG-Adriana-combined"
    labels = pd.read_csv(labels_file_path, sep='\s', header=None, engine='python')
    labels.columns = [
        "type",
        "index",
        "label"
    ]
    # We only use the ADC rows
    labels = labels[labels['type'] == "ADC"]
    # We don't need to know this anymore
    labels = labels.drop(["type"], axis=1)
    
    merged_data = pd.merge(data,labels, left_on='clip', right_on='index')
    merged_data.drop(["index","num", "time"], axis=1, inplace=True)
    return merged_data

def get_train_test():
    """reads in train and test data, in that order"""
    train_file_path = "data/algebra_replay_nums_model-training"
    test_file_path = "data/algebra_replay_nums_model-test"

    train = pd.read_csv(train_file_path, sep='\t', header=None)
    train.columns = [
        "index",
        "lesson_id"
    ]

    test = pd.read_csv(test_file_path, sep='\t', header=None)
    test.columns = [
        "index",
        "lesson_id"
    ]
    return train, test

def format_data(dummied_data, train, test):
    """takes in data, train and test sets and sorts data into train_x, train_y, test_x, test_y"""
    # create empty sets
    test_x = []
    train_x = []
    test_y = []
    train_y = []

    # loop to format data
    for clip, groupdf in dummied_data.groupby("clip"):
#     for clip, groupdf in tqdm(dummied_data.groupby("clip"), total=len(dummied_data.groupby("clip"))):
        # if correct len
        if len(groupdf.values) == 5:
            toplist = []
            ident = groupdf["clip"].values[0]

            if ident in list(test.iloc[:,0]):

                # get this clip's label
                label = groupdf["label"].values[0]
                if label == "N":
                    test_y.append(0)
                if label == "G":
                    test_y.append(1)

                # drop extra columns
                groupdf = groupdf.drop(columns=["clip", "label"])

                # adding clip to df
                for listindex in groupdf.values:
                    toplist.append(list(listindex))

                test_x.append(toplist)
            else:
                # get this clip's label
                label = groupdf["label"].values[0]
                if label == "N":
                    train_y.append(0)
                if label == "G":
                    train_y.append(1)

                # drop extra columns
                groupdf = groupdf.drop(columns=["clip", "label"])

                # adding clip to df
                for listindex in groupdf.values:
                    toplist.append(list(listindex))

                train_x.append(toplist)
    return train_x, train_y, test_x, test_y


def encode_actions(labeled_data):
    dummies = pd.get_dummies(labeled_data["action"])
    dummied_data = pd.concat([labeled_data.drop(columns=["action"]), dummies], axis=1)
    dummied_data = dummied_data.loc[dummied_data.loc[:, "label"] != "?"]
    return dummied_data

def get_data():
    """reads main feature file, does all formatting, results in lists: train_x, train_y, test_x, test_y"""
    df = pd.read_csv (r'data/featureOutput.csv')
    labeled_data = add_labels(df)
                      
    dummied_data = encode_actions(labeled_data)
    train, test = get_train_test()
    train_x, train_y, test_x, test_y = format_data(dummied_data, train, test)    
    
    return train_x, train_y, test_x, test_y

# import json
# import torch


# def load(course):
#     with open('data/'+course+'.json', "r") as content:
#         j = json.load(content)
#         data = []
#         for i in j:
#             s = [[], [], [], []]
#             norm = []
#             for k in j[i]:
#                 norm.append([l-1 for l in k])
#             s.extend(norm)
#             data.append({'seq': s, 'id': i})
#     return data


# def breakintosubpattern(data, k):
#     patterns = []
#     for i in data:
#         for j in range(len(i['seq'])-k-1):
#             patterns.append([i['seq'][j:j+k], i['seq'][j+k+1]])
#     return patterns


# def align(data):
#     num = [len(e['seq']) for e in data]
#     max_num = max(num)
#     id = [int(e['id']) for e in data]
#     batch_seq = []
#     for i in data:
#         s_ = i['seq'].copy()
#         s_.extend([[]]*(max_num-len(s_)))
#         batch_seq.append(to_onehot(s_))
#     batch_seq = torch.stack(batch_seq, dim=0)
#     return batch_seq, num, id


# def to_onehot(labels, n_categories=9, dtype=torch.float32):
#     batch_size = len(labels)
#     one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
#     for i, label in enumerate(labels):
#         label = torch.LongTensor(label)
#         one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
#     return one_hot_labels


# def colloate_fn(batch):
#     batch_seq = []
#     batch_truth = []
#     for i in batch:
#         s_ = i[0].copy()
#         batch_seq.append(to_onehot(s_))
#         s_t = i[1].copy()
#         batch_truth.append(to_onehot([s_t]))
#     batch_seq = torch.stack(batch_seq, dim=0)
#     batch_truth = torch.stack(batch_truth, dim=0)
#     return batch_seq, batch_truth
