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

# python get_sample_sets.py --tr_samples 5 --ts_samples 2 --docs 1500

parser = argparse.ArgumentParser(description='Prepare data for discourse detection')

parser.add_argument('--data_path', type=str, default='../data/sample_df_all.pkl', help='file containing data')
parser.add_argument('--emb_path', type=str, default='../data/embeddings_dict_all', help='file containing the embeddings')
parser.add_argument('--save_path', type=str, default='samples/', help='path to save results')
parser.add_argument('--tr_samples', type=int, default=6, help='Number of training datasets created and used')
parser.add_argument('--ts_samples', type=int, default=2, help='Number of test datasets created and used')
parser.add_argument('--num_timepoints', type=int, default=100, help='number of timepoints in each sample')
parser.add_argument('--min_docs', type=int, default=500, help='minimum number of documents in each datapoint')
parser.add_argument('--max_docs', type=int, default=1500, help='maximum number of documents in each datapoint')
parser.add_argument('--test_categories', type=list, default=['autot', 'musiikki', 'luonto', 'vaalit', 'taudit', 'työllisyys'], help='News categories used in the test data')
parser.add_argument('--tr_val_categories', type=list, default=['jääkiekko', 'kulttuuri', 'rikokset', 'koulut', 'tulipalot', 'ruoat'], help='News categories used in the train and validation data')
parser.add_argument('--n_topics_tr', type=int, default=6, help='number of topics used in the training samples')
parser.add_argument('--n_topics_ts', type=int, default=6, help='number of topics used in the test samples')

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
#subject_list = list(itertools.chain(*list(df['subjects'])))

# Count the occurrence of each topic (number of articles with the topic)
#subject_count = Counter(subject_list)

# Select categories for training and validation that are not in test categories and are not extremely rare
#tr_val_cats = [x[0] for x in subject_count.most_common() if x[1] > 8000 and x[0] not in args.test_categories]

#del subject_list
#del subject_count

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
tr_val_df = extract_categories(tr_val_data, args.tr_val_categories)
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


# Creates sample triples
def sample_triples(df, d_type='train', n_samples=100, min_doc=50, max_doc=100, timeline=100, change_rates=[0.5, 1]):
    
    categories = df['category'].unique()
    
    samples = []   
    tracker = pd.DataFrame(columns=['category', 'pattern', 'pivots'])
    unstable_patterns = ['up', 'down', 'up_down', 'down_up', 'spike_up', 'spike_down']
    all_patterns = ['up', 'down', 'up_down', 'down_up', 'spike_up', 'spike_down', 'stable']
    
    g = df.groupby(['category'])
    
    for _ in range(n_samples):
        # select random category as the target and the rest as noise
        stable_cats = np.random.choice(categories, 6, replace=False)
        unstable_cats = np.random.choice(stable_cats, 3, replace=False)
        stable_cats = [i for i in stable_cats if i not in unstable_cats]
        
        # Positive samples have the same unstable pattern and change rate
        pos_pattern = np.random.choice(unstable_patterns)
        pos_change_rate = np.random.uniform(*change_rates)
        pos_docs = np.random.randint(min_doc, max_doc)
        
        # Negative samples have different unstable pattern and randomly chosen change rate
        neg_patterns = [i for i in all_patterns if i != pos_pattern]
        neg_pattern = np.random.choice(neg_patterns)
        neg_change_rate = np.random.uniform(*change_rates)
        neg_docs = np.random.randint(min_doc, max_doc)
        
        freqs_unstable_pos, pivots_unstable_pos = sample_pattern(pos_pattern, timeline=timeline, change_rate=pos_change_rate)
        freqs_stable_pos, _ = sample_pattern('stable', timeline=timeline, change_rate=pos_change_rate)
        
        freqs_unstable_neg, pivots_unstable_neg = sample_pattern(neg_pattern, timeline=timeline, change_rate=neg_change_rate)
        freqs_stable_neg, _ = sample_pattern('stable', timeline=timeline, change_rate=neg_change_rate)
        
        sample_triple = []
        
        # Three sample sets are created in one loop
        for i in range(3):
            
            df_sample = []
            
            pattern = pos_pattern
            docs = pos_docs
            pivots = pivots_unstable_pos
            
            sample_cats = [unstable_cats[i], stable_cats[i]]
            
            if i == 2:
                pattern = neg_pattern
                docs = neg_docs
                pivots = pivots_unstable_neg
            
            for c in sample_cats:
                
                if c in unstable_cats and i != 2:
                    freqs = freqs_unstable_pos
                elif c in unstable_cats and i == 2:
                    freqs = freqs_unstable_neg      
                elif c in stable_cats and i != 2:
                    freqs = freqs_stable_pos
                elif c in stable_cats and i == 2:
                    freqs = freqs_stable_neg
            
                # get n_doc, which is random between min and max but not exceed the total docs in cluster
                df_cat = g.get_group(c)[['id', 'category']]

                n_doc = min(docs, len(df_cat)) 

                # calculate the docs_num based on the total docs and its freqs distribution
                docs_num = np.array(n_doc * freqs).astype(int)

                sample = df_cat.sample(n_doc)
                sample['time'] = -1

                # assign the sampled time points to the docs
                cur = 0
                for j, n in enumerate(docs_num):
                    sample.iloc[cur:cur+n, sample.columns.get_loc("time")] = j
                    cur += n

                # because the freq is converted to int, so the n_doc > docs_num. so some articles will remain -1 for time, we need to prunt those
                sample = sample[sample['time'] > -1]
                df_sample.append(sample)

            df_sample = pd.concat(df_sample, ignore_index=True)
            sample_triple.append(df_sample)
            tracker = tracker.append({'category': unstable_cats[i], 'pattern': pattern, 'pivots': pivots}, ignore_index=True)
        # Saves the samples as lists of three dataframes    
        samples.append(sample_triple)
        
    return samples, tracker


# Creates the labels for the timepoints based on the pivot points
def convert_pivots(tracker, timepoints):
    
    df = tracker.copy()
    df['labels'] = pd.Series(np.array)
    
    for i, row in df.iterrows():
        row['labels'] = np.zeros(timepoints).astype(int)
        
        if row['pattern'] == 'up_down' or row['pattern'] == 'down_up':
            row['pivots'] = np.array([row['pivots'][0], row['pivots'][2]])
            row['labels'][row['pivots'][0]:row['pivots'][1]+1] = 1
            
        elif row['pattern'] == 'spike_up' or row['pattern'] == 'spike_down':
            temp = []
            for p in row['pivots']:
                temp.append(p-2)
                temp.append(p+2)
            row['pivots'] = np.array(temp)
            
            for i in range(0,len(temp)-1, 2):
                row['labels'][temp[i]:temp[i+1]+1] = 1 

        elif row['pattern'] == 'stable':
            continue       

        else:
            row['labels'][row['pivots'][0]:row['pivots'][1]+1] = 1
               
    return df


# Adds the labels to the sample dataframes
def add_labels(samples, tracker_labels):

    new_samples = []
    ind = 0

    # Loop through the triplest    
    for triplet in samples:
        new_triplet = []
        
        # Loop through each triplet
        for df in triplet:
            data = df.copy()
            labels = []
            
            # Loop through dataframe rows
            for j, row in data.iterrows():
                timepoint = row['time']
                labels.append(tracker_labels['labels'][ind][timepoint])

            data['label'] = labels
            new_triplet.append(data)
            ind += 1 
           
        new_samples.append(new_triplet)
        
    return new_samples


# Creates the defined number of samples using categories based on the chosen type (train, validation, test)
def get_samples(d_type, n_samples):
    
    if d_type == 'test':
        df = test_df
    else:
        df = tr_val_df
    
    # Get samples
    samples, tracker = sample_triples(df, d_type=d_type, n_samples=n_samples, timeline=args.num_timepoints, min_doc=args.min_docs, max_doc=args.max_docs, change_rates=[0.5, 1])
    # Add timepoint labels to tracker
    tracker_labels = convert_pivots(tracker, args.num_timepoints)
    # Add timepoint labels to data
    samples = add_labels(samples, tracker_labels)

    return samples, tracker_labels

# Get the dosument embeddings corresponding to each article
def get_embeddings(dataframe):
    
    df = dataframe.copy()

    embeddings = {}
    labels = {}

    tensor_list = []
    label_list = []

    for i, row in df.iterrows():
        if row['id'] in embeddings_dict:
            emb = embeddings_dict[row['id']]
            timepoint = row['time']
            label = row['label']

            # Embeddings belonging to articles in same timepoint are stacked together
            if timepoint in embeddings:
                embeddings[timepoint] = np.vstack([embeddings[timepoint], emb])
            else:
                embeddings[timepoint] = emb

            if timepoint not in labels:
                labels[timepoint] = label

    # Embeddings of articles in the same sample set are conbined together as one tensor      
    for t in range(len(embeddings)):
        tensor = torch.from_numpy(embeddings[t])
        tensor_list.append(tensor)
        label_list.append(labels[t])

    # Adds padding to tensors that have less documents than the maximum number of docs in timepoint
    padded_data = pad_sequence(tensor_list)

    return padded_data, label_list


# Creates a dict based on the data sample where timepoints are keys and embeddings are the values
def get_data(d_type, n_samples):
    
    # Get dataset and tracker with timepoint labels
    samples, tracker_labels = get_samples(d_type, n_samples)

    # Variable for recording the maximum amount of documents per timepoint for padding
    max_docs = 0
        
    for ind, triplet in enumerate(samples):
        new_triplet = []
        new_labels = []
        
        # Gets the embeddings for each data sample in a triplet
        for df in triplet:
            data, labels = get_embeddings(df)

            if data.shape[0] > max_docs:
                max_docs = data.shape[0]
                
            new_triplet.append(data)
            new_labels.append(labels)

        # Saves each sample triplet as a dict with data tensor and labels tensor
        sample_dict = {'data': new_triplet[0], 'pos_pair': new_triplet[1], 'neg_pair': new_triplet[2], 'labels': new_labels} 
        output_file = open(args.save_path + d_type + "/triplets/%i.pkl"%ind, "wb")
        pickle.dump(sample_dict, output_file)
        output_file.close()        
        
    return max_docs, tracker_labels


max_docs = 0

# Get test data
max_docs_ts, ts_tracker = get_data('test', args.ts_samples)

print(ts_tracker)

del test_df
del ts_tracker

# Get training data
max_docs_tr, tr_tracker = get_data('train', args.tr_samples)

# get the maximum number of docs per timepoint in all data
if max_docs_ts > max_docs_tr:
    max_docs = max_docs_ts
else:
    max_docs = max_docs_tr

print(tr_tracker)

print("Max size: ", max_docs)

# Save max number of docs to be used in padding
f = open(args.save_path + "max_docs_triple.txt", "w")
f.write(str(max_docs))
f.close()