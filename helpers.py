import pandas as pd
import regex as re
from os import listdir
from os.path import isfile, join
import numpy as np

def read(DATA_PATH):
    Dataset = {}
    filenames = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
    for filename in filenames:
        if filename[-4:] == '.txt':
            print(filename)
            with open(DATA_PATH + '/' + filename, 'r') as f:
                lines = f.read().splitlines() 
                Dataset[filename[:-4]] = lines
    return Dataset

def re_sub(pattern, repl, text):
    return re.sub(pattern, repl, text, count = len(re.findall(pattern, text)))
def clean_text(tweet):

    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = re_sub(r"<3","<heart>", tweet)
    tweet = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", tweet)
    tweet = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ", tweet)
    tweet = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ", tweet)
    tweet = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ", tweet)
    tweet = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", tweet)
    tweet = re_sub(r"/"," / ", tweet)
    tweet = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", tweet)
    tweet = re_sub(r"([!?.]){2,}", r"\1 <repeat> ", tweet)
    tweet = re_sub(' +', ' ', tweet)


    tweet = tweet.replace('won\'t', ' will not')
    tweet = tweet.replace('dunno', 'do not know')
    tweet = tweet.replace('n\'t', ' not')
    tweet = tweet.replace('i\'m', 'i am')
    tweet = tweet.replace('\'re', ' are')
    tweet = tweet.replace('it\'s', 'it is')
    tweet = tweet.replace('that\'s', 'that is')
    tweet = tweet.replace('\'ll', ' will')
    tweet = tweet.replace('\'l', ' will')
    tweet = tweet.replace('\'ve', ' have')
    tweet = tweet.replace('\'d', ' would')
    tweet = tweet.replace('he\'s', 'he is')
    tweet = tweet.replace('what\'s', 'what is')
    tweet = tweet.replace('who\'s', 'who is')
    tweet = tweet.replace('\'s', '')
    return tweet
    
def clean(Dataset):
    
    for key in Dataset.keys():
        Dataset[key] = pd.DataFrame(Dataset[key])
        Dataset[key] = Dataset[key].applymap(clean_text)
        if key != 'test_data':
            Dataset[key] = remove_repeated(Dataset[key])

    return Dataset
    

    return Dataset
def remove_repeated(datarow):
    datarow.drop_duplicates(inplace = True )
    return datarow

def save_to_file(Dataset, DATA_PATH=''):
    for key in Dataset.keys():
        np.savetxt(DATA_PATH+ '/' + str(key) + '_clean.txt', Dataset[key].values, fmt='%s')
def read_test_file(filename):
    """
    DESCRIPTION: 
            Reads a file and returns it as a list
    INPUT: 
            filename: Name of the file to be read
    """
    data = []
    with open(filename, "r", encoding='utf8') as ins:
        for line in ins:
          line = line.partition(',')[2]
          #line = line.split(' ')
          data.append(line.strip())
    return data
def read_data(dataset):
    """
    DESCRIPTION: 
            reads training data from the files and stores them into dataframes
    INPUT: 
            data_path: the directory path to the data files
    RETURN: 
            train_pos: df with the positive training tweets
            train_neg: df with the negative training tweets
            test_data: df with the test tweets
    """
    train_pos = pd.DataFrame(dataset['train_pos_clean'])
    train_pos = train_pos.applymap(lambda x: x.strip())
    train_pos = train_pos[0].str.split(' ', expand = True)
    train_pos['sentiment'] = 1
    train_neg =  pd.DataFrame(dataset['train_neg_clean'])
    train_neg = train_neg.applymap(lambda x: x.strip())
    train_neg = train_neg[0].str.split(' ', expand = True)
    train_neg['sentiment'] = 0
    test_data =  pd.DataFrame(dataset['test_data_clean'])
    test_data = test_data.applymap(lambda x: x.partition('>')[2])
    test_data = test_data.applymap(lambda x: x.strip())
    
    test_data = test_data[0].str.split(' ', expand = True)

    return train_pos, train_neg, test_data

def read_data_full(dataset):
    """
    DESCRIPTION: 
            reads training data from the files and stores them into dataframes
    INPUT: 
            data_path: the directory path to the data files
    RETURN: 
            train_pos: df with the positive training tweets
            train_neg: df with the negative training tweets
            test_data: df with the test tweets
    """
    train_pos = pd.DataFrame(dataset['train_pos_full_clean'])
    train_pos = train_pos.applymap(lambda x: x.strip())
    train_pos = train_pos[0].str.split(' ', expand = True)
    train_pos['sentiment'] = 1
    train_neg =  pd.DataFrame(dataset['train_neg_full_clean'])
    train_neg = train_neg.applymap(lambda x: x.strip())
    train_neg = train_neg[0].str.split(' ', expand = True)
    train_neg['sentiment'] = 0
    test_data =  pd.DataFrame(dataset['test_data_clean'])
    test_data = test_data.applymap(lambda x: x.partition('>')[2])
    test_data = test_data.applymap(lambda x: x.strip())
    
    test_data = test_data[0].str.split(' ', expand = True)

    return train_pos, train_neg, test_data

def findembedding(word, glove_embd):
    model_input = glove_embd[word]
    return model_input

def find_token(word, vocab):
    if word in vocab:
        return vocab[word]
    else:
        return 5


