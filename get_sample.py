import pickle
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm

import torch
from torch.nn.utils.rnn import pad_sequence

# python get_sample.py --num_samples 5  --min_docs 200 --max_docs 700

parser = argparse.ArgumentParser(description='Prepare data for discourse detection')

parser.add_argument('--data_path', type=str, default='data/filtered_12cats_lemmas.pkl', help='file containing data')
parser.add_argument('--emb_path', type=str, default='data/embeddings_dict', help='file containing the embeddings')
parser.add_argument('--save_path', type=str, default='data/samples/', help='path to save results')
parser.add_argument('--num_samples', type=int, default=4, help='number of created samples')
parser.add_argument('--num_timepoints', type=int, default=100, help='number of timepoints in each sample')
parser.add_argument('--min_docs', type=int, default=500, help='minimum number of documents in each datapoint')
parser.add_argument('--max_docs', type=int, default=1500, help='maximum number of documents in each datapoint')

args = parser.parse_args()
print(args)

# Load the dataset
df = pd.read_pickle(args.data_path)

# Load the embedding data in dictionary form
with open(args.emb_path, 'rb') as handle:
    embeddings_dict = pickle.load(handle)

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

def create_test_samples(df, n_samples=100, min_doc=50, max_doc=100, frac=0.99, timeline=100, change_rates=[0.5, 1]):
    categories = df['category'].unique()
    
    samples = []   # list article ids
    tracker = pd.DataFrame(columns=['category', 'pattern', 'pivots'])
    # sample_pivots = []  # list of pivots index in timeline, need to map with ids
    patterns = ['up', 'down', 'up_down', 'down_up', 'spike_up', 'spike_down']
    # events = np.random.choice(patterns, n_samples, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    # patterns = ['spike_down']
    
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
            df_cat = g.get_group(c)[['id', 'category', 'body']]
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

# Creates a dict based on the data sample where timepoints are keys and 
# embeddings are the values
def get_embeddings(data, emb_dict):
    
    tensor_list = []
    label_list = []
    
    for dataframe in data:
        
        df = dataframe.copy()

        embeddings = {}
        labels = {}

        for i, row in df.iterrows():
            emb = emb_dict[row['id']]
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
        
        
    return tensor_list, torch.FloatTensor(label_list)


# Creates a list of data samples and tracker that shows the discourse patterns
test_samples, tracker = create_test_samples(df, n_samples=args.num_samples, timeline=args.num_timepoints, min_doc=args.min_docs, max_doc=args.max_docs, frac=0.99, change_rates=[0.5, 1])

del df

print(tracker)

# Uses the tracker to create the labels (stable/non-stable) for each timepoint in the samples
tracker_labels = convert_pivots(tracker, args.num_timepoints)

# Adds the labels to the sample files
samples = add_labels(test_samples, tracker_labels)

# Creates tensors and labels
tensors, labels = get_embeddings(samples, embeddings_dict)

# Adds padding to tensors that have less documents than the maximum number of docs in timepoint
padded_data = pad_sequence(tensors)

# Save the created samples and the tracker
pickle.dump(samples, open(args.save_path + "samples_5.pkl", "wb"))
pickle.dump(tracker_labels, open(args.save_path + "tracker_labels_5.pkl", "wb"))

# Save the tensor with the padded data and the labels
d = {'data': padded_data, 'labels': labels}
torch.save(d, args.save_path + 'tensors_5.pt')