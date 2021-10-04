import pandas as pd
import os

import numpy as np
import tqdm

import gensim
from gensim import models
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import random

from pprint import pprint
import pickle 


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as stop_words

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')


speeches = pd.read_csv('./all_ECB_speeches.csv', delimiter='|', error_bad_lines=False)
speeches.head()

#Remove NA entries
speeches = speeches.dropna()

#Only get presidential speeches
speeches = speeches.loc[speeches.subtitle.str.contains("\sPresident\s"),:]


#Regex cleaning
speeches['contents'] = speeches['contents'].replace('SPEECH', '', regex=True)
speeches['contents'] = speeches['contents'].replace('\((.*?)\)', '', regex=True)
speeches['contents'] = speeches['contents'].replace('\[(.*?)\]', '', regex=True)
speeches['contents'] = speeches['contents'].replace('Note.*?\.', '', regex=True)
speeches['contents'] = speeches['contents'].replace('Chart .*?\..*?\.', '', regex=True)
speeches['contents'] = speeches['contents'].replace('[,\.!?]', '', regex=True)
speeches['contents'] = speeches['contents'].replace('\s[a-z]{1,2}\s', '', regex=True)
speeches['contents'] = speeches['contents'].replace('[^\x00-\x7F]+',' ', regex=True)
speeches['contents'] = speeches['contents'].replace('[^\w\s]', '', regex=True)


# stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
words = set(nltk.corpus.words.words())


# preprocessing functions
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_non_english(texts):
    return [[w for w in nltk.wordpunct_tokenize(" ".join(doc)) if w.lower() in words or not w.isalpha()] for doc in texts]

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def lemmatize(texts):
    return [[lemmatizer.lemmatize(w) for w in doc] for doc in texts]

def noun_only(texts):
    return [[word[0] for word in nltk.pos_tag(doc) if word[1] in ['NN','JJ','JJR','JJS','NNP','NNS']] for doc in texts]



def preprocess(input_data):
    data = input_data.contents.values.tolist()


    # data = [input_data.iloc[1].contents]

    data_words = list(sent_to_words(data))


    data_words = remove_non_english(data_words)
    
    data_words = remove_stopwords(data_words)
    data_words = lemmatize(data_words)

    data_words = remove_stopwords(data_words)

    data_words = noun_only(data_words)

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    data_words = make_bigrams(data_words)

    return data_words

def gen_corpus(data_words):
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # id2word.filter_extremes( no_above=0.9, keep_n=100000)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    return id2word, corpus, corpus_tfidf


def run_lda(id2word, corpus, data_words, valid_corpus, k=5,  a='symmetric', b=None, coherence_type="u_mass"):
    
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word, 
                                        workers=20, 
                                        num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    
    coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence=coherence_type)
    

    return lda_model, coherence_model_lda.get_coherence(), 2**(-lda_model.log_perplexity(valid_corpus))



## Data Generation

# Topics range
topics_range = range(2, 20, 1)
print(len(topics_range))

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

print(len(alpha))

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

print(len(beta))

# Range Type
range_of_data = ["entire"]
# range_of_data = ["entire", "yearly", "quarterly"]
print(len(range_of_data))

# Range
entire = [(0, "1997-2021")]
yearly = [(i, str(i)) for i in range(1997, 2021)]
quarters = ["(1|2|3)","(4|5|6)", "(7|8|9)","(10|11|12)"]
quarterly = [[((year, quarter), str(year)+"_"+str(idx+1)) for idx,quarter in enumerate(quarters)] for year in range(1997, 2021)]
print(len(entire))

# Coherence Type
coherence_types = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
print(len(coherence_types))

# Corpus Type
corpus_types = ['bow','tfidf']
print(len(corpus_types))

# Validation sets
corpus_title = list(np.arange(0.1, 0.9, 0.2))
print(len(corpus_title))

def corpus_sets(corpus):
    temp = []
    num_of_docs = len(corpus)
    for percentage in corpus_title:
        random.shuffle(corpus)
        temp.append((corpus[:round(percentage*num_of_docs)],corpus[round(percentage*num_of_docs):]))
    return temp




# data_words = preprocess(speeches)

# id2word, corpus, corpus_tfidf = gen_corpus(data_words)

# model, cv, p = run_lda(id2word, corpus_tfidf, data_words)

model_results = {
                 'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Range_Type':[],
                 'Range':[],
                 'Coherence_Type': [],
                 'Corpus_Type': [],
                 'Coherence': [],
                 'Perplexity':[],
                }

    # len(entire)+len(yearly)+len(quarterly)
total = len(topics_range) * len(alpha) * len(beta) * (1) * len(coherence_types) * len(corpus_types) * len(corpus_title)



# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=540)


    
    # iterate through number of topics
    for k in topics_range:
        # iterate through alpha values
        for a in alpha:
            # iterare through beta values
            for b in beta:
                # iterate through bow , tfidf
                data = speeches
                data_words = preprocess(data)
                id2word, corpus, corpus_tfidf = gen_corpus(data_words)
                for corpus_type in corpus_types:
                    all_corpus = None
                    if corpus_type == "bow":
                        all_corpus = corpus
                    elif corpus_type == "tfidf":
                        all_corpus = corpus_tfidf

                    # iterate through validation corpuses
                    for idx, (train_corpus, valid_corpus) in enumerate(corpus_sets(all_corpus)):

                        for coherence_type in coherence_types:
                            # get the coherence score for the given parameters
                            print()
                            print("Range: 1997-2021, k: {}, a: {}, b: {}, corpus_type:{}, coherence_type:{},validation_percent:{}".format(k, a, b, corpus_type, coherence_type, corpus_title[idx]))
                            model, cv, p = run_lda(id2word, train_corpus, data_words, valid_corpus, k=k, a=a, b=b, coherence_type=coherence_type)
                            print("Coherence measure: {}, Perplexity: {}".format(cv, p))
                
                            # Save the model results
                            model_results['Validation_Set'].append(corpus_title[i])
                            model_results['Topics'].append(k)
                            model_results['Alpha'].append(a)
                            model_results['Beta'].append(b)
                            model_results['Coherence_Type'].append(coherence_type)
                            model_results['Corpus_Type'].append(corpus_type)
                            model_results['Range_Type'].append('entire')
                            model_results['Range'].append('1997-2021')

                            model_results['Coherence'].append(cv)
                            model_results['Perplexity'].append(p)
                            
              
            
                            pbar.update(1)


    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()