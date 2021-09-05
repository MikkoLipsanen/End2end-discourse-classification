import pickle
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm
from collections import Counter
import itertools
import random

import torch
from torch.nn.utils.rnn import pad_sequence

# python get_sample.py --num_samples 5  --min_docs 200 --max_docs 700

parser = argparse.ArgumentParser(description='Prepare data for discourse detection')

parser.add_argument('--data_path', type=str, default='data/sample_df_all.pkl', help='file containing data')
parser.add_argument('--emb_path', type=str, default='data/embeddings_dict_all', help='file containing the embeddings')
parser.add_argument('--save_path', type=str, default='data/samples/', help='path to save results')
parser.add_argument('--tr_samples', type=int, default=6, help='Number of training datasets created and used')
parser.add_argument('--ts_samples', type=int, default=2, help='Number of test datasets created and used')
parser.add_argument('--num_timepoints', type=int, default=100, help='number of timepoints in each sample')
parser.add_argument('--min_docs', type=int, default=500, help='minimum number of documents in each datapoint')
parser.add_argument('--max_docs', type=int, default=1500, help='maximum number of documents in each datapoint')
parser.add_argument('--test_categories', type=list, default=['autot', 'musiikki', 'luonto', 'vaalit', 'taudit', 'työllisyys', 'jääkiekko', 'kulttuuri', 'rikokset', 'koulut', 'tulipalot', 'ruoat'], help='News categories used in the test data')
parser.add_argument('--n_topics_tr', type=int, default=12, help='number of topics used in the training samples')
parser.add_argument('--n_topics_ts', type=int, default=12, help='number of topics used in the test samples')


args = parser.parse_args()
print(args)

############Load data##############

# Load the dataset with article ids and topics
df = pd.read_pickle(args.data_path)

# Load the embedding data in dictionary form
with open(args.emb_path, 'rb') as handle:
    embeddings_dict = pickle.load(handle)


######Split data based on topics#####

# Create a list of all topics in the dataset
subject_list = list(itertools.chain(*list(df['subjects'])))

# Count the occurrence of each topic (number of articles with the topic)
subject_count = Counter(subject_list)

# Select categories for training and validation that are not in test categories and are not extremely rare
tr_val_cats = [x[0] for x in subject_count.most_common() if x[1] > 8000 and x[0] not in args.test_categories]

del subject_list
del subject_count

# Split dataset into 2 sets based on the given categories used in the test data
# Train and validation datasets are sampled from articles that don't contain any of the test categories
def split_dataset(df, cats):
    
    cat_set = set(cats)
    is_cat = df['subjects'].apply(lambda x: (cat_set & set(x) == set()))
    tr_val_data = df[is_cat].copy()
    test_data = df[~is_cat].copy()
    
    return tr_val_data, test_data

# Selects one category 'label' for each article
def extract_categories(df, cats):
    clusters = []
    
    for cat in cats:
        is_cat = df['subjects'].apply(lambda x: (cat in x))
        df_filtered = df[is_cat].copy()
        df_filtered['category'] = cat
        clusters.append(df_filtered)
        
    df_merged = pd.concat(clusters, ignore_index=True)
    df_merged = df_merged.drop(columns=['subjects'])
    
    return df_merged


# Split dataset into two parts so that articles in train and validation sets don't contain categories chosen for the test set
tr_val_data, test_data = split_dataset(df, args.test_categories)

# Select one "representative" category for each article 
tr_val_df = extract_categories(tr_val_data, tr_val_cats)
test_df = extract_categories(test_data, args.test_categories)

del tr_val_data
del test_data

#############Sample datasets###################

# Sampling patterns for the synthetic data
def linear_pattern(n=1, start=0, stop=100, change_rate=1):
    """
    Sampling up pattern, start and end in random points
    """
    x = np.arange(start, stop)
    # normalize x to range 0-1
    y = (x - start) / (stop - start)
    freq_rates = n + y * n * change_rate
    
    return freq_rates

def sigmoid_pattern(n=1, start=0, stop=100, change_rate=1):
    x = np.arange(start, stop)
    mid = int((stop - start) / 2)
    y = 1 / (1 + np.exp(-0.1* (x-mid) ))
    y = (y - y.min()) / (y.max() - y.min())
    
    freq_rates = n + y * n * change_rate
    return freq_rates

def flat_pattern(n=1, start=0, stop=100):
    freq_rates = np.ones(stop-start) * n
    return freq_rates

def bell_pattern(n=1, start=0, stop=100, change_rate=1, std=0):
    sample_list = []
    time_range = stop - start
    
    x = np.arange(start, stop)
    mu = int(time_range / 2)
    
    std = std if std else int(time_range / 5)
    y = norm.pdf(np.arange(time_range), mu, std)
    # scale 0-1
    y = (y - y.min()) / (y.max() - y.min())
    # add n docs
    freq_rates = n + y * n * change_rate
    
    return freq_rates


def sample_pattern(pattern, timeline=100, change_rate=0.01):
    sample = None
    
    if pattern == 'up':
        lower_p = np.random.randint(low=1, high=timeline-30)
        upper_p = np.random.randint(low=lower_p+20, high=timeline)
        
        # f1, f2, f3 [-1] is the start of freqs ratio for the pattern as the chaning variable
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = sigmoid_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        change_points = np.array([lower_p, upper_p])
        
    elif pattern == 'down':
        lower_p = np.random.randint(low=1, high=timeline-30)
        upper_p = np.random.randint(low=lower_p+20, high=timeline)
        
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = sigmoid_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=-change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        change_points = np.array([lower_p, upper_p])
        
    elif pattern == 'spike_up':
        n_point = np.random.randint(1, 5)
        invalid = True
        
        while invalid:
            change_points = np.sort(np.random.choice(range(5, timeline - 5), n_point, replace=False))
            diff = np.diff(change_points)
            invalid = len(np.where(diff < 10)[0])
            
        change_rates = np.random.uniform(0.3, change_rate, n_point)
        cur_p = 0
        cur_n = 1
        
        time_freqs = []
        
        for i, p in enumerate(change_points):
            #print(cur_p, p - 2)
            f1 = flat_pattern(cur_n, start=cur_p, stop=p-2)
            cur_n = f1[-1]
            f2 = bell_pattern(cur_n, start=p-2, stop=p+3, change_rate=change_rates[i], std=0.1)
            cur_n = f2[-1]
            
            time_freqs.append(f1)
            time_freqs.append(f2)
            
            cur_p = p + 3
            
            if i == len(change_points) - 1:
                f3 = flat_pattern(cur_n, start=cur_p, stop=timeline)
                time_freqs.append(f3)

        time_freqs = np.concatenate(time_freqs)
        time_freqs = time_freqs / time_freqs.sum()
        
    elif pattern == 'spike_down':
        n_point = np.random.randint(1, 5)
        invalid = True
        # generate n points with min distance 10
        while invalid:
            change_points = np.sort(np.random.choice(range(5, timeline - 5), n_point, replace=False))
            diff = np.diff(change_points)
            invalid = len(np.where(diff < 10)[0])
            
        change_rates = np.random.uniform(0.3, change_rate, n_point)
        cur_p = 0
        cur_n = 1
        
        time_freqs = []
        
        for i, p in enumerate(change_points):

            f1 = flat_pattern(cur_n, start=cur_p, stop=p-2)
            cur_n = f1[-1]
            f2 = bell_pattern(cur_n, start=p-2, stop=p+3, change_rate=-change_rates[i], std=0.1)
            cur_n = f2[-1]
            
            time_freqs.append(f1)
            time_freqs.append(f2)
            
            cur_p = p + 3
            
            if i == len(change_points) - 1:
                f3 = flat_pattern(cur_n, start=cur_p, stop=timeline)
                time_freqs.append(f3)
            
        time_freqs = np.concatenate(time_freqs)
        time_freqs = time_freqs / time_freqs.sum()
        
    elif pattern == 'up_down':
        lower_p = np.random.randint(low=1, high=timeline-20)
        upper_p = np.random.randint(low=lower_p+10, high=timeline)
        
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = bell_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()
        
        mid_p = int(lower_p + (upper_p - lower_p) / 2)
        change_points = np.array([lower_p, mid_p, upper_p])
        
    elif pattern == 'down_up':
        lower_p = np.random.randint(low=1, high=timeline-20)
        upper_p = np.random.randint(low=lower_p+10, high=timeline)
        
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = bell_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=-change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        mid_p = int(lower_p + (upper_p - lower_p) / 2)
        change_points = np.array([lower_p, mid_p, upper_p])
        
    else:
        time_freqs = flat_pattern(1, start=0, stop=timeline)
        time_freqs = time_freqs / time_freqs.sum()
        change_points = np.empty(shape=(0,))
        
    return time_freqs, change_points.astype(int)

# Creates the data samples
def create_samples(df, d_type='train', n_samples=100, min_doc=50, max_doc=100, frac=0.99, timeline=100, change_rates=[0.5, 1]):

    unique_categories = list(df['category'].unique())
    
    if d_type == 'test':
        categories = random.sample(unique_categories, args.n_topics_ts)
    elif d_type == 'train':
        categories = random.sample(unique_categories, args.n_topics_tr)
    else:
        print("Select one of the following sample data types: 'test', 'train'")
    
    samples = []   # list article ids
    tracker = pd.DataFrame(columns=['category', 'pattern', 'pivots'])
    patterns = ['up', 'down', 'up_down', 'down_up', 'spike_up', 'spike_down']
    
    g = df.groupby(['category'])
    
    for _ in range(n_samples):
        # select random category as the target
        # And the rest as noise
        cat = np.random.choice(categories)
        pattern = np.random.choice(patterns)
        change_rate = np.random.uniform(*change_rates)
        df_sample = []
        
        for c in categories:
            if c == cat:
                freqs, pivots = sample_pattern(pattern, timeline=timeline, change_rate=change_rate)
            else:
                freqs, _ = sample_pattern('stable', timeline=timeline, change_rate=change_rate)
        
            # get n_doc, which is random between min and max but not exceed the total docs in cluster
            df_cat = g.get_group(c)[['id', 'category']]
            df_len = len(df_cat)
            n_doc = np.random.randint(min_doc, max_doc)
            n_doc = min(n_doc, df_len)
        
            # calculate the docs_num based on the total docs and its freqs distribution
            docs_num = (n_doc * freqs).astype(int)
            sample = df_cat.sample(n_doc)
            sample['time'] = -1
            
            # assign the sampled time points to the docs
            cur = 0
            for i, n in enumerate(docs_num):
                sample.iloc[cur:cur+n, sample.columns.get_loc("time")] = i
                cur += n

            # because the freq is converted to int, so the n_doc > docs_num. so some articles will remain -1 for time, we need to prunt those
            sample = sample[sample['time'] > -1]
            df_sample.append(sample)
            
        df_sample = pd.concat(df_sample, ignore_index=True)
        #df_sample = df_sample.sample(frac=frac)
        samples.append(df_sample)
        
        tracker = tracker.append({'category':cat, 'pattern': pattern, 'pivots': pivots}, ignore_index=True)
        
    return samples, tracker

# Creates the lables for the timepoints based on the pivot points
def convert_pivots(tracker, timepoints):
    
    df = tracker.copy()
    df['labels'] = pd.Series(np.array)
    
    for i, row in df.iterrows():
        row['labels'] = np.zeros(timepoints).astype(int)
        
        if row['pattern'] == 'up_down' or row['pattern'] == 'down_up':
            row['pivots'] = np.array([row['pivots'][0], row['pivots'][2]])
            row['labels'][row['pivots'][0]:row['pivots'][1]] = 1
            
        elif row['pattern'] == 'spike_up' or row['pattern'] == 'spike_down':
            temp = []
            for p in row['pivots']:
                temp.append(p-2)
                temp.append(p+2)
            row['pivots'] = np.array(temp)
            
            for i in range(0,len(temp)-1, 2):
                row['labels'][temp[i]:temp[i+1]] = 1             
        else:
            row['labels'][row['pivots'][0]:row['pivots'][1]] = 1
               
    return df

# Adds the labels to the sample dataframes
def add_labels(test_samples, tracker_labels):

    samples = []

    for i, df in enumerate(test_samples):
        data = df.copy()
        labels = []
        for j, row in data.iterrows():
            timepoint = row['time']
            labels.append(tracker_labels['labels'][i][timepoint])
            
        data['label'] = labels
        samples.append(data)
        
    return samples

# Creates the defined number of samples using categories based on the chosen type (train, validation, test)
def get_samples(d_type, n_samples):
    
    if d_type == 'test':
        df = test_df
    else:
        df = tr_val_df
        
    test_samples, tracker = create_samples(df, d_type=d_type, n_samples=n_samples, timeline=args.num_timepoints, min_doc=args.min_docs, max_doc=args.max_docs, frac=0.99, change_rates=[0.5, 1])
    
    tracker_labels = convert_pivots(tracker, args.num_timepoints)
    
    samples = add_labels(test_samples, tracker_labels)
    
    return samples, tracker_labels

# Creates a dict based on the data sample where timepoints are keys and embeddings are the values
def get_embeddings(d_type, n_samples):
    
    samples, tracker_labels = get_samples(d_type, n_samples)
    
    tensor_list = []
    label_list = []
    
    for dataframe in samples:
        
        df = dataframe.copy()

        embeddings = {}
        labels = {}

        for i, row in df.iterrows():
            if row['id'] in embeddings_dict:
                emb = embeddings_dict[row['id']]
                timepoint = row['time']
                label = row['label']
                if timepoint in embeddings:
                    embeddings[timepoint] = np.vstack([embeddings[timepoint], emb])
                else:
                    embeddings[timepoint] = emb

                if timepoint not in labels:
                    labels[timepoint] = label
           
        for t in range(len(embeddings)):
            tensor = torch.from_numpy(embeddings[t])
            tensor_list.append(tensor)
            label_list.append(labels[t])
            
    # Adds padding to tensors that have less documents than the maximum number of docs in timepoint
    padded_data = pad_sequence(tensor_list)
        
    return padded_data, torch.FloatTensor(label_list), tracker_labels

# Variable for recording the maximum amount of documents per timepoint for padding
max_docs = 0

# Get test data
ts_tensor, ts_labels, ts_tracker = get_embeddings('test', args.ts_samples)

print("Test data shape", ts_tensor.shape)

if ts_tensor.shape[0] > max_docs:
    max_docs = ts_tensor.shape[0]

print(ts_tracker)

ts_d = {'data': ts_tensor, 'labels': ts_labels}
torch.save(ts_d, args.save_path + 'ts_data.pt')

del test_df
del ts_tensor
del ts_labels
del ts_tracker

# Get training data
tr_tensor, tr_labels, tr_tracker = get_embeddings('train', args.tr_samples)

print("Training data shape", tr_tensor.shape)

if tr_tensor.shape[0] > max_docs:
    max_docs = tr_tensor.shape[0]

print(tr_tracker)

print("Max size: ", max_docs)

# Save the tensor with the padded data and the labels
tr_d = {'data': tr_tensor, 'labels': tr_labels, 'max_size': max_docs}
torch.save(tr_d, args.save_path + 'tr_data.pt')

