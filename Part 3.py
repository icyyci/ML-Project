#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
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


# In[2]:


import copy
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
def generate_prediction(input_file_path, output_file_path, labels, transition_table, emission_table, word_list):
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
            label_out = viterbi(sentence, labels, transition_table, emission_table, word_list)
            labels_out.append(label_out)
            
    with open(output_file_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(words_copy))):
            for j in range(len(words_copy[i])):
                output = words_copy[i][j] + " " + labels_out[i][j] + "\n"
                fout.write(output)
            fout.write("\n")


# In[3]:


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
                    if i+1 == len(sentence):
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


# In[4]:


#To get YX's emission table
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


# In[5]:


'''
Inputs:
n (list)： list of input words to be predicted
l_labels (str)： list of all possible states
tr_arr (2d numpy array): transition table
em_arr (2d numpy array): emission table
word_list (list): list of words

Output:
args (list): list of predicted states 

Function:
Takes input sentence of n words, outputs list of n predicted tags/states corresponding to input words 
'''
def viterbi(n, l_labels, tr_arr, em_arr, word_list):
        
    # Initializoing pi array 
    pi_arr = np.zeros([len(l_labels[1:-1]),len(n)+2])
    
    ### Forwarding Algo
    
    # Iterating through the columns of the pi array
    for j in range(len(n)+2):
        
        # column 0: START nodes, assign value 1 to all nodes in column 0
        if j == 0:
            for i in range(len(pi_arr)):
                pi_arr[i][j] = 1
                
        # column 1: nodes right after START, pi value = 1(pi value of START) * transition_prob * emission_ prob       
        elif j == 1:
            if n[j-1] not in word_list:
                n[j-1] = "#UNK#"
            for u_idx, u in enumerate((l_labels[1:-1])):
                if tr_arr[0][u_idx] == 0 or em_arr[word_list.index(n[j-1])][u_idx] == 0:
                    pi_arr[u_idx][j] = float('-inf')
                else:
                    pi_arr[u_idx][j] = np.log(tr_arr[0][u_idx]) + np.log(em_arr[word_list.index(n[j-1])][u_idx])
        
        # columns 2 to n : pi value = max(pi value of previous nodes * transition_prob * emission_ prob)
        elif j > 1 and j < len(n)+1:
            
            # n[j-1]: current word in sentence, if not found in word_list, replace with "#UNK#"
            if n[j-1] not in word_list:
                n[j-1] = "#UNK#"
            
            # Iterating through column v: current column
            for u_idx, u in enumerate((l_labels[1:-1])): # v
                
                # array to store temp scores
                pi_temp = []
                
                # Iterating through column u: previous column
                for u_idx_temp, u_temp in enumerate((l_labels[1:-1])): # u
                    if tr_arr[u_idx_temp+1][u_idx] == 0 or em_arr[word_list.index(n[j-1])][u_idx] == 0:
                        pi_temp.append(float('-inf'))
                    else:
                        # append pi_value_v (current) = pi_value_u (previous) * transition_prob(u,v) * emission_prob(v,word)  
                        pi_temp.append(pi_arr[u_idx_temp][j-1] + np.log(tr_arr[u_idx_temp+1][u_idx]) + np.log(em_arr[word_list.index(n[j-1])][u_idx]))
                
                #pi_value_v = max(_temp)
                pi_arr[u_idx][j] = max(pi_temp) 
                
        # column n+1 : STOP node: pi value = max(pi value of previous nodes * transition_prob)
        elif j == len(n)+1:
            pi_temp = []
            for u_idx, u in enumerate((l_labels[1:-1])):
                if tr_arr[u_idx+1][len(l_labels[1:-1])] == 0:
                    pi_temp.append(float("-inf"))
                else:
                    pi_temp.append(np.log(tr_arr[u_idx+1][len(l_labels[1:-1])]) + pi_arr[u_idx][j-1])
            for u_idx_temp, u_temp in enumerate((l_labels[1:-1])):
                pi_arr[u_idx_temp][j] = max(pi_temp)
                
    ### Backtracking Algo     
    
    # list to store predicted outputs
    args = []
    
    # To store the index current node with the highes score
    last_idx = len(l_labels[1:-1])
    
    # Iterating from n to 1: n, n-1, n-2...1
    for i in range(len(n),0,-1):
        
        # array to store all temp scores calculated 
        temp = []
        
        # Iterating through the rows
        for u_idx, u in enumerate((l_labels[1:-1])):
            if tr_arr[u_idx+1][last_idx] == 0:
                temp.append(float("-inf"))
            else:
                # append the score = transition_prob * pi value to temp
                temp.append(np.log(tr_arr[u_idx+1][last_idx]) + pi_arr[u_idx][i])
                
        # update last_idx with the index of the node that had the highest score
        # if all the scores are "-inf", output label "O"
        if np.max(temp) == float("-inf"):
            last_idx = 7
        else:
            last_idx = np.argmax(temp)
        
        # append tag/label corresponding to node with highest score to output
        args.append(l_labels[last_idx+1])

    return list(reversed(args))
    


# In[6]:


words_en, labels_en = train("EN/train")

transition_table_en = generate_transition_table(words_en, labels_en)

x_en, y_en = clean("EN/train")
word_list_en = generate_word_list(x_en)
tag_word_table_en = generate_table(x_en, y_en)
emission_table_en = generate_emission_table(x_en, word_list_en, tag_word_table_en)

generate_prediction("EN/dev.in", "EN/dev.p3.out", labels_en, transition_table_en, emission_table_en, word_list_en)

print("EN donez")

words_cn, labels_cn = train("CN/train")

transition_table_cn = generate_transition_table(words_cn, labels_cn)

x_cn, y_cn = clean("CN/train")
word_list_cn = generate_word_list(x_cn)
tag_word_table_cn = generate_table(x_cn, y_cn)
emission_table_cn = generate_emission_table(x_cn, word_list_cn, tag_word_table_cn)

generate_prediction("CN/dev.in", "CN/dev.p3.out", labels_cn, transition_table_cn, emission_table_cn, word_list_cn)

print("CN donez")

words_sg, labels_sg = train("SG/train")

transition_table_sg = generate_transition_table(words_sg, labels_sg)

x_sg, y_sg = clean("SG/train")
word_list_sg = generate_word_list(x_sg)
tag_word_table_sg = generate_table(x_sg, y_sg)
emission_table_sg = generate_emission_table(x_sg, word_list_sg, tag_word_table_sg)

generate_prediction("SG/dev.in", "SG/dev.p3.out", labels_sg, transition_table_sg, emission_table_sg, word_list_sg)

print("SG donez")


# In[ ]:




