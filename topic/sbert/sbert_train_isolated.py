# define this if you have more than 1 gpu
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
import numpy as np
import pickle


def kl_divergence(p,q):
    return np.sum(p * (np.log2(p)-np.log2(q)))

def js_divergence(p,q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def one_hot(p,q):
    return 1 if p==q else 0

def sigmoid(val):
    # return val
    return 1/(1+np.exp(-17*(val-0.5)))

def identity(val):
    return val

# js ensure symmetric
def similarity(p,q,mode="js", func=identity):
    if mode == "js":
        return func(np.exp2(-js_divergence(np.array(p),np.array(q))))
    elif mode == "kl":
        return func(np.exp2(-kl_divergence(np.array(p),np.array(q))))
    elif mode == "one-hot":
        return one_hot(p,q)

def get_random_index_pairs(num_data, amount):
    return np.random.randint(num_data, size=(amount, 2))


# flatten to one list for all 3
with open('train_data.pickle', 'rb') as file:
    train = pickle.load(file)

with open('gpt.pickle', 'rb') as file:
    gpt = pickle.load(file)
    
with open('gpt_p2.pickle', 'rb') as file:
    gpt2 = pickle.load(file)

gpt = [item for sublist in gpt for item in sublist]
gpt2 = [item for sublist in gpt2 for item in sublist]

mixed = gpt + gpt2
test = train

print(len(mixed))
print(len(test))

from itertools import combinations
import random
all_pairs = list(combinations(range(len(mixed)),2))

random.shuffle(all_pairs)
# bert load data
data = [{"texts":[mixed[idx[0]]["text"],mixed[idx[1]]["text"]], "label": similarity(mixed[idx[0]]["dist"],mixed[idx[1]]["dist"])} for idx in all_pairs]


train, dev = data[:1500000],data[1500000:1505000]

pair1 = [item["texts"][0] for item in dev]
pair2 = [item["texts"][1] for item in dev]
scores = [float(item["label"]) for item in dev]

#Define your train examples. You need more than just two examples...
train_examples = [InputExample(texts=item["texts"], label=float(item["label"])) for item in train]

print(len(train_examples))


#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
# model = SentenceTransformer('./sbert', device='cuda')

evaluator = evaluation.EmbeddingSimilarityEvaluator(pair1, pair2, scores)


#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator)

model.save("./res/sbert_v2","sbert_v2")