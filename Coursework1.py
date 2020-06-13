import json
import re
import nltk
import math
import random
from nltk import *
from nltk.corpus import wordnet
from collections import defaultdict

with open("signal-news1.jsonl", 'r') as f:
    # Read .json file and Lowercase the text
    text = [json.loads(line.lower()[:]) for line in f]

with open("positive-words.txt") as f:
    positive_words = f.read()

with open("negative-words.txt") as f:
    negative_words = f.read()


# Define a function to parse and clean the texts by using regular expressions
def parse_and_clean(pattern, signal_news1):
    text_remove = list(filter(None, [re.sub(pattern, r'', str(line)) for line in signal_news1]))
    return text_remove


# Step1. Remove the URL
pattern1 = re.compile(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|(?:%[0-9a-zA-Z][0-9a-zA-Z]))+')
text_remove1 = parse_and_clean(pattern1, text)

# Step2 Remove non-alphanumeric characters and spaces by regular expression ([^\sa-zA-Z0-9])
pattern2 = re.compile(r'[^\sa-zA-Z0-9]')
text_remove2 = parse_and_clean(pattern2, text_remove1)

# Step3 Remove words with only one character
pattern3 = re.compile(r'(\b[a-zA-Z]\b)')
text_remove3 = parse_and_clean(pattern3, text_remove2)

# Step4 Remove numbers that are fully made of digits
pattern4 = re.compile(r'(\b\d+\b)')
text_remove4 = parse_and_clean(pattern4, text_remove3)

# Q1.2
# Use pos_tagger to tag word in news
tagged = []

for items in text_remove4:
    tagged.append(pos_tag(word_tokenize(items)))


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


wnl = WordNetLemmatizer()
lemma = []
for lists in tagged:
    for n in range(len(lists)):
        wordnet_pos = get_wordnet_pos(lists[n][1]) or wordnet.NOUN
        lists[n] = list(lists[n])  # Change the tuple into list in order to make changes
        lists[n][0] = wnl.lemmatize(lists[n][0], pos=wordnet_pos)  # Lemmatize the words according to POS tagging
        lemma.append(wnl.lemmatize(lists[n][0], pos=wordnet_pos))  # Append the lemmatized words into a lemma list
        lists[n].pop()  # Remove the tag
        lists[n] = ''.join(lists[n])  # Convert the type of words from list to strings

print("Here are the lemmatized words: ", lemma)


# Part B: N-grams
# Q1. Number of tokens and vocabulary size

N = len(lemma)
V = len(set(lemma))

print("Number of tokens (N) is ", N)
print("Vocabulary size (V) is ", V)


# Q2. List trigrams and sort it by number of occurrences
trigram = nltk.trigrams(lemma)
tri_freqdist = nltk.FreqDist(trigram)
print("Here is the list of top 25 trigrams: \n", tri_freqdist.most_common(25))


# Q3. Compute the number of positive and negative word counts in the corpus
# Create a frequency distribution of an uni-gram
uni_freqdist = nltk.FreqDist(lemma)

positive_words_list = positive_words.splitlines()
negative_words_list = negative_words.splitlines()

# Count
num_positive = 0
num_negative = 0

for k, v in uni_freqdist.items():
    if k in positive_words_list[:]:
        num_positive += 1

    if k in negative_words_list[:]:
        num_negative += 1

print("Number of positive word counts in the corpus: ", num_positive)
print("Number of negative word counts in the corpus: ", num_negative)


# Q4. Compute the number of stories after comparison

# Define a function to count the positive words and negative words in each sublist
def count_pos_neg(item):
    i = 0
    pos = 0
    neg = 0
    while i < len(item):
        if item[i] in positive_words_list[:]:
            pos += 1

        if item[i] in negative_words_list[:]:
            neg += 1

        i += 1

    return [pos, neg]


num_more_positive_news = 0
num_more_negative_news = 0

# Compare the number of positive words and negative words
for lists in tagged:
    count = count_pos_neg(lists)

    if count[0] > count[1]:
        num_more_positive_news += 1

    if count[0] < count[1]:
        num_more_negative_news += 1

print("Number of positive news stories: ", num_more_positive_news)
print("Number of negative news stories: ", num_more_negative_news)


# Part C: Language Models
# Q1. Compute language models for trigrams and

train_tagged = tagged[:16000]
test_tagged = tagged[16000:]

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of co-occurrence
for lists in train_tagged:
    for w1, w2, w3 in trigrams(lists, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1


# Number of the unique bi-grams (V), which will be added in the denominator
add_v_train = []
for lists in train_tagged:
    bigram_train = bigrams(lists)
    for items in bigram_train:
        if items not in add_v_train:
            add_v_train.append(items)

# Transforms the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        numerator = model[w1_w2][w3] + 1
        denominator = total_count + len(add_v_train)
        model[w1_w2][w3] *= int(math.log(numerator) - math.log(denominator))  # Transform the probability into log


# Produce a sentence of 10 words by appending the most likely next word each time
text = ["be", "this"]
EOS = False  # End-of-Sentence
while not EOS:
    dic = sorted(dict(model[tuple(text[-1:])]).items(), key=lambda d: d[1], reverse=True)
    print("dict:", dic)
    
    text.append(dic[0])
    if len(text) == 10:
        EOS = True
print(' '.join([t for t in text if t]))


# Q2. Compute the perplexity
pi = 0
perplexity = 0

for lists in test_tagged:
    for w1, w2, w3 in trigrams(lists, pad_right=True, pad_left=True):
        for w1_w2 in model:
            for w3 in model[w1_w2]:
                pi += model[w1_w2][w3]
                perplexity = ((1 / pi) ** len(lists))

print(perplexity)
