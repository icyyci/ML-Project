#!/usr/bin/env python
# coding: utf-8

# ### Part 4

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import copy


# In[2]:


'''
Inputs:
file_path (str): path to input file

Output:
words (3d numpy array): list of sentences;
                        each sentence: list of words coupled with corresponding state 
                        [[[word, state],[word_2, state_2],...],[sentence_2],...]
labels (list): list of tags/states including START and STOP

Function:
Takes input file, outputs the words arranged with corresponding states and a list of all possible tags/states
'''
def train(file_path):
    with open (file_path, 'r', encoding="utf-8") as f: 
        lines = f.readlines()
        words = []
        labels = []

        temp = []
        for l in lines: 
            if l != "\n":
                l_split = l.strip().split(" ")
                te = " ".join(l_split[:-1])
                la = l_split[-1]
                temp.append([te, la])
                labels.append(l_split[-1])
            else: 
                words.append(temp)
                temp = []

    l_labels = ["START"]
    for i in labels:
        if i in l_labels:
            continue
        else:
            l_labels.append(i)
    l_labels.append("STOP")
    return words, l_labels

'''
Inputs:
u (str)： initial state/tag
v (str)： final state/tag
words (list)：list of words

Output:
transition_probability (float): transition probability from state u to state v

Function:
Takes initial state u and final state v, calculates the transition probability from state u to state v
'''
def transition(u,v,words):
    count_u = 0
    count_u_v = 0
    for sentence in words:
        if u == "START":
            count_u = len(words)
            if sentence[0][1] == v:
                count_u_v +=1
        elif v == "STOP":
            for i in range(len(sentence)):
                if sentence[i][1] == u:
                    count_u += 1
                    if i+1==len(sentence):
                        count_u_v += 1
        else:
            for i in range(len(sentence)):
                if sentence[i][1] == u:
                    count_u += 1
                    if i+1!=len(sentence) and sentence[i+1][1] == v:
                        count_u_v += 1
    return count_u_v/count_u

'''
Inputs:
words (list)：list of words
l_labels (list): list of all possible states

Output:
transition_table (2d numpy array): transition table for given training set 

Function:
Takes list of all possible states and outputs the transition table for given training set
'''
def generate_transition_table(words, l_labels):
    transition_table = np.zeros([len(l_labels[:-1]),len(l_labels[1:])])
    for row_idx, label_row in tqdm(enumerate(l_labels[:-1])):
        for col_idx, label_col in enumerate(l_labels[1:]):
            transition_table[row_idx][col_idx] = transition(label_row, label_col,words)
    return transition_table


# In[3]:


'''
Inputs:
x (list): list of words with corresponding tag in y
y (list): list of tags with corresponding word in x
list_of_tags (list): all tags that could be formed

Output:
tag_word_table (dictionary): dictionary with keys being tags and values being the associated words

Function:
Format data into a dictionary. Length of x and y must be the same.
'''
def generate_table(x, y):
    if len(x)!=len(y):
        print("ERROR: difference in length between data and tag")
        return None
    
    tag_word_table = {}
    for tag, word in tqdm(zip(y,x)):
        if tag in tag_word_table:
            tag_word_table[tag].append(word)
        else:
            tag_word_table[tag] = [word]
            
    return tag_word_table

'''
Inputs:
x (str): word to be queried
y (str): label to be queried
tag_word_table (dictionary): data in table form
k (float): number of occurences #UNK# is found

Output:
probability (float): probability of generating x from y based on tag_word_table

Function:
Calculated emission probability
'''
def emission(x, y, tag_word_table, k = 0.5):
    word_list = tag_word_table[y]
    if x == "#UNK#":
        emission_count = k
    else:
        emission_count = word_list.count(x)
    ycount = len(word_list)
    return emission_count / (ycount + k)

'''
Inputs:
x (list): list of words

Output:
word_list (list): list of unique words

Function:
Generates a list of all unique words in x
'''
def generate_word_list(x):
    word_list = []
    for i in tqdm(x):
        if i not in word_list:
            word_list.append(i)
    word_list.append("#UNK#")
    return word_list

'''
Inputs:
list_x (list): list of words
word_list (list): list of unique words
tag_word_table (dictionary): dictionary form of the data

Output:
emission_table (numpy array): 2D numpy array with row each row representing a word and each column a tag

Function:
Generates the emission table, where each word has its emission value stored in a numpy array
'''
def generate_emission_table(list_x, word_list, tag_word_table):
    # Each row is the word
    # Each column is the tag in tag_word_table
    emission_table = np.zeros([len(word_list), len(tag_word_table.keys())])
    
    tags = tag_word_table.keys()
    
    for ind_x,x in tqdm(enumerate(word_list)):
        for ind_y, y in enumerate(tags):
            em = emission(x,y,tag_word_table)
            emission_table[ind_x,ind_y] = em
    return emission_table

'''
Inputs:
input_file (str): path to input file

Output:
x_list (list): list of words
y_list (list): list of tags

Function:
Cleans the input file, as some lines have spaces within the word section. This function only takes the last
word delimited by spaces as the tag then recombines all words delimited by space in front to form the actual
word. Then returns the data as a list of word_list and tag_list.
'''
def clean(input_file):
    inp_f = open(input_file, "r", encoding="utf-8")
    lines = inp_f.readlines()
    x_list = []
    y_list = []
    for ind, l in tqdm(enumerate(lines)):
        words = l.split(" ")
        if len(words)>2:
            tag = words[-1].strip("\n")
            act_word = " ".join(words[:-1])
            x_list.append(act_word)
            y_list.append(tag)
        elif len(words)==2:
            x_list.append(words[0])
            y_list.append(words[1].strip("\n"))
        elif len(words)==1:
            continue
        else:
            print(words)
            print(str(ind) + "training data has no label")
            print("data is: " + words[0])
    return x_list, y_list


# In[4]:


'''
Input:
prev_pi_vals (len_statesx3x2 numpy array): previous pi values
len_states (int): num of possible states
tr_arr (2d numpy arr): transition table
em_arr (2d numpy arr): emission table
curr_state (int): current state index in the table
word_idx (int): index of current word in word_list

Output:
top3 (3x2 numpy array): array of the top 3 values with the 1st value being the highest

Function:
Finds the top 3 probabilities and their parents
'''
def find_top3(prev_pi_vals, len_states, tr_arr, em_arr, curr_state, word_idx):
    # Initialize
    temp_vals = []

    for u_idx, state in enumerate(prev_pi_vals):
        for pi_val, parent_idx in state:
            prev_state = u_idx
            tr_val = tr_arr[prev_state+1][curr_state]
            em_val = em_arr[word_idx][curr_state]
            if tr_val == 0 or em_val == 0:
                new_pi_val = float('-inf')
            else:                        
                new_pi_val = pi_val + np.log(tr_val) + np.log(em_val)
            temp_vals.append([new_pi_val, prev_state])
            
    # Sort the values in descending order by using the first value of the list (pi_val)
    if np.max(temp_vals) == float("-inf"):
        top3 = np.array([[float("-inf"), 7], [float("-inf"), 7], [float("-inf"), 7]])
    else:
        sorted_vals = sorted(temp_vals, key = lambda x: x[0], reverse=True)
        top3 = np.array(sorted_vals[:3])

    return top3

'''
Input:
parent_idx (int): index of parent of STOP
n (list): input sentence
pi_arr (2d numpy arr): table of pi values
all_states (list): list of all the tags not including START and STOP
paths_table (nested list): table containing all the paths of previous iterations for each node

Output:
pred_vals (list): predicted values
paths_table (list of list): updated path table for this iteration

Function:
Traceback using the parent values
'''
                                      
def traceback(parent_idx, n, pi_arr, all_states, paths_table):
    pred_vals = []
    path = ['STOP']
    for curr_idx in range(len(n)-1,0,-1):
        best_idx = paths_table[curr_idx][parent_idx].count(path)
        paths_table[curr_idx][parent_idx].append(path)
        
        next_path = path + [all_states[parent_idx]]
        path = next_path
        pred_vals.append(all_states[parent_idx])

        _, parent_idx = pi_arr[parent_idx, curr_idx, best_idx, :]
        parent_idx = int(parent_idx)
        
    pred_vals.append(all_states[parent_idx])
    pred_vals.reverse()
    return pred_vals, paths_table

'''
Input:
n (list of str): sentence split into list by spaces
l_labels (list of str): list of tags
tr_arr (2d numpy arr): transmission table tr_arr[initial state, final state]
em_arr (2d numpy arr): emission table em_arr[word, state]
word_list (list of str): list of all words in the training set

Output:
pred_tag (list of string): list of predicted tags

Function:
Does modified viterbi algorithm to find the top 3 best path
'''

def viterbi_top3(n, l_labels, tr_arr, em_arr, word_list):
    args = []
    all_states = l_labels[1:-1]
    len_states = len(all_states)
    
    # Set up pi array with each state containing top 3 pi states and within each state contains the [pi_prob, parent_idx]
    pi_arr = np.zeros([len_states,len(n),3,2])
    
    # going through the table
    for j in range(len(n)):
        
        curr_word = n[j]
        if curr_word not in word_list:
            curr_word = "#UNK#"
        word_idx = word_list.index(curr_word)
        
        # First column
        if j == 0:
            for curr_state in range(len_states):
                if tr_arr[0][curr_state] == 0 or em_arr[word_idx][curr_state] == 0:
                    pi_val = float('-inf')
                else:
                    pi_val = np.log(tr_arr[0][curr_state]) + np.log(em_arr[word_idx][curr_state])
                pi_arr[curr_state,j] =  np.array([[pi_val, 0], [pi_val, 0], [pi_val, 0]])
                
        # Second column
        elif j == 1:
            for curr_state in range(len_states):
                temp_vals = []
                prev_pi_vals = pi_arr[:,j-1,:,:]
                for u_idx, state in enumerate(prev_pi_vals):
                    pi_val, parent_idx = state[0,:]
                    prev_state = u_idx
                    if tr_arr[prev_state+1][curr_state] == 0 or em_arr[word_idx][curr_state] == 0:
                        new_pi_val = float('-inf')
                    else:
                        new_pi_val = pi_val + np.log(tr_arr[prev_state+1][curr_state]) + np.log(em_arr[word_idx][curr_state])
                    temp_vals.append([new_pi_val, prev_state])
                # Sort the values in descending order by using the first value of the list (pi_val)
                if np.max(temp_vals) == float("-inf"):
                    pi_arr[curr_state,j] = np.array([[float("-inf"), 7], [float("-inf"), 7], [float("-inf"), 7]])
                else:
                    sorted_vals = sorted(temp_vals, key = lambda x: x[0], reverse=True)
                    pi_arr[curr_state,j] = np.array(sorted_vals[:3])
    
        # All other columns
        elif j > 1:
            for curr_state in range(len_states):
                prev_pi_vals = pi_arr[:,j-1,:,:]
                top3 = find_top3(prev_pi_vals, len_states, tr_arr, em_arr, curr_state, word_idx)
                pi_arr[curr_state,j] = top3
    
    temp_vals = []
    last_pi_vals = pi_arr[:,len(n)-1,:,:]
    for u_idx, state in enumerate(last_pi_vals):
        for pi_val, parent_idx in state:
            prev_state = u_idx
            tr_val = tr_arr[prev_state+1][curr_state]
            if tr_val == 0:
                new_pi_val = float('-inf')
            else:
                new_pi_val = pi_val + np.log(tr_val)
            temp_vals.append([new_pi_val, prev_state])
    # Sort the values in descending order by using the first value of the list (pi_val)
    if np.max(temp_vals) == float("-inf"):
        top3 = np.array([[float("-inf"), 7], [float("-inf"), 7], [float("-inf"), 7]])
    else:
        sorted_vals = sorted(temp_vals, key = lambda x: x[0], reverse=True)
        top3 = np.array(sorted_vals[:3])
    
    # paths_table contains all the paths that visited that node for each node in the graph
    # the path at index j and with state_ind state is paths_table[j][state] 
    paths_table = []
    path = []
    for j in range(len(n)):
        temp_paths = []
        for state in range(len_states):
            temp_paths.append([])
        paths_table.append(temp_paths)
    
    pred_vals_best, paths_table = traceback(int(top3[0,1]), n, pi_arr, all_states, paths_table)
    pred_vals_2ndbest, paths_table = traceback(int(top3[1,1]), n, pi_arr, all_states, paths_table)
    pred_vals_3rdbest, paths_table = traceback(int(top3[2,1]), n, pi_arr, all_states, paths_table)
    
    return pred_vals_best


# In[5]:


'''
Inputs:
input_file_path (str): path to input file
output_file_path (str): path to output file
labels (list): list of all possible tags/states
transition_table (2d numpy array): transition table calculated from training set
emission_table (2d numpy array): emission table calculated from training set
word_list (list): list of words

Output:
 - : write predicted results to output file

Function:
Takes input file, for every sentence in input file, generates the predicted tag/state for each word in sentence by running
viterbi() and write to output file 
'''
def generate_prediction_top3(input_file_path, output_file_path, labels, transition_table, emission_table, word_list):
    with open (input_file_path, 'r', encoding="utf-8") as dev_in: 
        lines = dev_in.readlines()
        words = []

        temp = []
        for l in lines: 
            if l != "\n":
                l_split = l.strip().split(" ")
                la = l_split[-1]
                temp.append(la)
            else: 
                words.append(temp)
                temp = []

        words_copy = copy.deepcopy(words)
        labels_out = []
        for sentence in tqdm(words):
            label_out = viterbi_top3(sentence, labels, transition_table, emission_table, word_list)
            labels_out.append(label_out)
            
    with open(output_file_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(words_copy))):
            for j in range(len(words_copy[i])):
                output = words_copy[i][j] + " " + labels_out[i][j] + "\n"
                fout.write(output)
            fout.write("\n")


# In[6]:


words_en, labels_en = train("EN/train")

transition_table_en = generate_transition_table(words_en, labels_en)

x_en, y_en = clean("EN/train")
word_list_en = generate_word_list(x_en)
tag_word_table_en = generate_table(x_en, y_en)
emission_table_en = generate_emission_table(x_en, word_list_en, tag_word_table_en)

print("EN donez")


# In[7]:


generate_prediction_top3("EN/dev.in", "EN/dev.p4.out", labels_en, transition_table_en, emission_table_en, word_list_en)

