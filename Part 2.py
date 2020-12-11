#!/usr/bin/env python
# coding: utf-8

# ### Part 2 of ML Project

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm


# In[2]:


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


# ### Gather Data

# In[3]:


# Training data for EN
# df_en = pd.read_csv("en/train", delim_whitespace=True, names = ["Word", "Tag"])
x_en, y_en = clean("en/train")

# Training data for SG
# df_sg = pd.read_csv("SG/train", delim_whitespace=True, names = ["Word", "Tag"])
x_sg, y_sg = clean("SG/train")

# Training data for CN
# df_cn = pd.read_csv("CN/train", delim_whitespace=True, names = ["Word", "Tag"])
x_cn, y_cn = clean("CN/train")


# In[4]:


#Word list for EN
word_list_en = generate_word_list(x_en)

#Word list for SG
word_list_sg = generate_word_list(x_sg)

#Word list for CN
word_list_cn = generate_word_list(x_cn)


# In[5]:


# Tag word table for EN
tag_word_table_en = generate_table(x_en, y_en)

# Emission table for EN
emission_table_en = generate_emission_table(x_en, word_list_en, tag_word_table_en)

# Tag word table for SG
tag_word_table_sg = generate_table(x_sg, y_sg)

# Emission table for SG
emission_table_sg = generate_emission_table(x_sg, word_list_sg, tag_word_table_sg)

# Tag word table for CN
tag_word_table_cn = generate_table(x_cn, y_cn)

# Emission table for CN
emission_table_cn = generate_emission_table(x_cn, word_list_cn, tag_word_table_cn)


# ### Find tag with highest emission and write to output file

# In[6]:


'''
Inputs:
x (str): word to be queried
emission_table (numpy array): table of emission values
word_list (list): list of unique words
tag_word_table (dictionary): dictionary form of input data

Output:
tag (str): tag with the highest probability for the input x

Function:
Finds the tag with the highest probability given the input x and the emission table
'''
def find_max(x, emission_table, word_list, tag_word_table):
    if x not in word_list:
        x = "#UNK#"
    prob_list = emission_table[word_list.index(x),:]
    max_ind = np.argmax(prob_list)
    return list(tag_word_table.keys())[max_ind]


# In[7]:


'''
Inputs:
input_file (str): path to input file
output_file (str): path to output file
emission_table (numpy array): table of emission values
word_list (list): list of unique words
tag_word_table (dictionary): dictionary form of input data

Output:
None

Function:
Finds the tag with the highest probability for each line in the input file and write it to the output file.
'''
def generate_pred(input_file, output_file, emission_table, word_list, tag_word_table):
    inp_f = open(input_file, "r", encoding="utf-8")
    out_f = open(output_file, "w", encoding="utf-8")
    inp_lines = inp_f.readlines()
    for x in tqdm(inp_lines):
        inp = x.strip("\n")
        if inp == "":
            out_f.write("\n")
            continue
        tag = find_max(inp, emission_table, word_list, tag_word_table)
        output = inp + " " + tag +"\n"
        out_f.write(output)
    inp_f.close()
    out_f.close()


# In[8]:


# Predict for input data from en
generate_pred("en/dev.in", "en/dev.p2.out", emission_table_en, word_list_en, tag_word_table_en)

# # Predict for input data from en
generate_pred("SG/dev.in", "SG/dev.p2.out", emission_table_sg, word_list_sg, tag_word_table_sg)

# # Predict for input data from en
generate_pred("CN/dev.in", "CN/dev.p2.out", emission_table_cn, word_list_cn, tag_word_table_cn)

