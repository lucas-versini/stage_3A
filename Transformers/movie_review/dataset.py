import pandas as pd
import numpy as np

import string
import re
import nltk
from nltk.corpus import stopwords

from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords')

class IMDBDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def one_hot_sentiment(data):
    """
    Convert the sentiment column to one-hot encoding
    """
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    return data

def clean_review(data):
    """
    Clean the review column (lowercase, remove URL, remove punctuation, remove multiple whitespaces, remove stop words, etc.)
    """
    
    data['review_clean'] = data['review'].str.lower()

    data['review_clean'] = data['review_clean'].apply(lambda x: re.sub(r'<.*?>', '', x)) # Remove HTML tags

    data['review_clean'] = data['review_clean'].apply(lambda x: re.sub(r"http\S+", "", x)) # Remove URL
    data["review_clean"] = data["review_clean"].apply(lambda x: x.replace("\n", " ")) # Remove new line
    data['review_clean'] = data['review_clean'].str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # remove punctuation

    data['review_clean'] = data['review_clean'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x)) # remove non alphabetic characters

    data['review_clean'] = data['review_clean'].str.strip() # remove leading and trailing whitespaces
    data['review_clean'] = data['review_clean'].apply(lambda x: re.sub(' +', ' ', x)) # remove multiple whitespaces

    stop_words = set(stopwords.words('english'))
    data['review_clean'] = data['review_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])) # remove stop words

    return data

def load_data(batch_size = 32, path = "./IMDB.csv"):
    # Load the data
    data = pd.read_csv(path)

    # Split the data into train/test
    data_train = data.sample(frac = 0.7, random_state = 100)
    data_test = data.drop(data_train.index)

    # Apply the functions to the data
    data_train = one_hot_sentiment(data_train)
    data_train = clean_review(data_train)

    data_test = one_hot_sentiment(data_test)
    data_test = clean_review(data_test)

    # Get the number of unique words and the maximum length of a review
    words_train = ' '.join(data_train['review_clean']).split()
    words_test = ' '.join(data_test['review_clean']).split()
    all_words = list(set(words_train + words_test))

    max_len = max([len(x.split()) for x in data_train['review_clean']] + [len(x.split()) for x in data_test['review_clean']])

    print("Number of unique words: ", len(all_words))
    print("Maximum length of a review: ", max_len)

    all_words.append("<PAD>")

    # Create an array that contains the word index of each review
    dic = {}
    for i, word in enumerate(all_words):
        dic[word] = i

    data_train["word_index"] = data_train["review_clean"].apply(lambda x: [dic[word] for word in x.split()])
    data_test["word_index"] = data_test["review_clean"].apply(lambda x: [dic[word] for word in x.split()])

    train_array_word_index = np.ones((len(data_train), max_len), dtype=int) * (len(all_words) - 1)
    for i in range(len(data_train)):
        train_array_word_index[i, :len(data_train["word_index"].iloc[i])] = data_train["word_index"].iloc[i]

    test_array_word_index = np.ones((len(data_test), max_len), dtype=int) * (len(all_words) - 1)
    for i in range(len(data_test)):
        test_array_word_index[i, :len(data_test["word_index"].iloc[i])] = data_test["word_index"].iloc[i]
    
    # Create the DataLoader
    data_loader_train = DataLoader(IMDBDataset(train_array_word_index, data_train["sentiment"].values), batch_size = batch_size, shuffle = True)
    data_loader_test = DataLoader(IMDBDataset(test_array_word_index, data_test["sentiment"].values), batch_size = batch_size, shuffle = False)

    return data_loader_train, data_loader_test, len(all_words), max_len, data_train, data_test, dic


    




