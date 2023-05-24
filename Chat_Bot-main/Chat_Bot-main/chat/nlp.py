import nltk
from nltk import sem
from nltk.stem.porter import PorterStemmer
from nltk.tag.brill import Word
from nltk.translate.meteor_score import wordnetsyn_match
import numpy as np

stemmer =PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1

    return bag 


# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

# print(bag_of_words(sentence,words))



