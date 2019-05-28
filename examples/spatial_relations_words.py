import tensorflow as tf
import logictensornetworks as ltn
import numpy as np
from numpy.random import choice as random
from logictensornetworks.operators import And, Or, Implies, Not, Equiv, Forall, Exists
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import gensim
import random
import os

agents = ['ben', 'greg', 'sarah', 'abi', 'alex']
movements = ['went', 'arrived', 'went']
locations = ['kitchen', 'hallway', 'office', 'campus']

# get all combinations of args
all_combinations = []
for combination_tuple in itertools.product(agents, movements, locations):
    all_combinations.append(' '.join(combination_tuple))
# take 80% of samples for training
n_train = int(len(all_combinations)*0.8)

# use most combinations for training but make sure some are omitted from test
# this is important to evaluate the systematicity and compositionality of learning
random.shuffle(list(set(all_combinations))) # take unique values and shuffle
train_combinations = all_combinations[:n_train]
test_combinations = all_combinations[n_train:]

assert all(c not in train_combinations for c in test_combinations), \
    "Test set should not contain sentences seen during training"

# Functions to generate data for training
def is_actor(substr, string):
    # Substring refers to actor if it is the first term
    return True if string.find(substr) == 0 else False

def is_movement(substr, string):
    # Substring refers to movement if it is the second term
    # Note: movements must be single words for this to work
    return True if string.split(' ')[1] == substr else False

def is_location(substr, string):
    # Substring refers to destination if it is the final term
    return True if string.split(' ')[-1] == substr else False

# get word2vec model from google pretrained
def get_embedding_data(sentence, model):
    words = sentence.split(' ')
    w2v = np.zeros((len(words), model["is"].shape[0]))
    for i,word in enumerate(words):
        w2v[i] = model[word]
    return w2v

# Load pretrained embedding weights
embedding_model_file = '../embeddings/GoogleNews-vectors-negative300.bin'
embedding_model_file_src = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit'
assert os.path.isfile(embedding_model_file), \
    "No embedding file found at '{}', please download from here: {}".format(embedding_model_file, embedding_model_file_src)

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_model_file, binary=True)

emb_size = w2v_model.vector_size

# generate positive and negative samples for each unary predicate

actor_data = np.array([get_embedding_data(w, w2v_model)[0]
              for s in train_combinations for w in s.split() if is_actor(w,s)])

not_actor_data = np.array([get_embedding_data(w, w2v_model)[0]
              for s in train_combinations for w in s.split() if not is_actor(w,s)])

movement_data = np.array([get_embedding_data(w, w2v_model)[0]
              for s in train_combinations for w in s.split() if is_movement(w,s)])

not_movement_data = np.array([get_embedding_data(w, w2v_model)[0]
              for s in train_combinations for w in s.split() if not is_movement(w,s)])

location_data = np.array([get_embedding_data(w, w2v_model)[0]
                       for s in train_combinations for w in s.split() if is_location(w, s)])

not_location_data = np.array([get_embedding_data(w, w2v_model)[0]
                           for s in train_combinations for w in s.split() if not is_location(w, s)])

all_words_data = np.array([get_embedding_data(w, w2v_model)[0]
                           for s in train_combinations for w in s.split()])

# and of data generations

# start the definition of the language:

# variables for pairs of rectangles ....

# ... for positive examples of every relation
act = ltn.variable("actor_data", tf.cast(actor_data, tf.float32))
mov = ltn.variable("movement_data",tf.cast(movement_data,tf.float32))
loc = ltn.variable("below_xy",tf.cast(location_data,tf.float32))

# ... for negative examples (they are placeholders which are filled with data
# randomly generated every 100 trian epochs

nact = ltn.variable("not_actor_data", emb_size)
nmov = ltn.variable("not_movement_data", emb_size)
nloc = ltn.variable("not_location_data", emb_size)


# printing out the dimensions of examples
pxy = [act, mov, loc]
npxy = [nact, nmov, nloc]

for xy in pxy:
    print(xy.name,xy.shape)

# variables for single embeddings

w = ltn.variable("w",emb_size)

# # some more constants and tensors to show results after training

ct = ltn.constant("ct", actor_data[0].astype('float32'))
t = ltn.variable("t",tf.cast(actor_data[1:], tf.float32))


# relational predicates

A = ltn.Predicate("actor", emb_size*2).ground
M = ltn.Predicate("movement", emb_size*2).ground
L = ltn.Predicate("location", emb_size*2).ground

P = [A,M,L]

n_pred = len(P)
# inv_P = [R,L,A,B,I,C]

# constraints/axioms

constraints =  [Forall(pxy[i],P[i](pxy[i]))
                for i in range(n_pred)]
constraints += [Forall(npxy[i],Not(P[i](npxy[i])))
                for i in range(n_pred)]
# constraints += [Forall((x,y),Implies(P[i](x,y),inv_P[i](y,x)))
#                 for i in range(6)]
# constraints += [Forall((x,y),Not(And(P[i](x,y),P[i](y,x))))
#                 for i in range(6)]
# constraints += [Forall((x,y,z),Implies(I(x,y),Implies(P[i](y,z),P[i](x,z)))) for i in range(6)]


loss = -tf.reduce_min(tf.concat(constraints,axis=0))
opt = tf.train.AdamOptimizer(0.05).minimize(loss)
init = tf.global_variables_initializer()


# generations of data for negative examples and generic rectangles used to feed the variables x,y,z

nr_random_words = 10
def get_feed_dict():
    feed_dict = {}
    feed_dict[nact] = not_actor_data[np.random.choice(len(not_actor_data), nr_random_words,replace=True)].astype(np.float32)
    feed_dict[nmov] = not_movement_data[np.random.choice(len(not_movement_data),  nr_random_words,replace=True)].astype(np.float32)
    feed_dict[nloc] = not_location_data[np.random.choice(len(not_location_data),nr_random_words,replace=True)].astype(np.float32)
    feed_dict[w] = all_words_data[np.random.choice(len(all_words_data),nr_random_words,replace=True)].astype(np.float32)
    return feed_dict


with tf.Session() as sess:

# training:

    sess.run(init)
    feed_dict = get_feed_dict()
    for i in range(1000):
        sess.run(opt,feed_dict=feed_dict)
        if i % 100 == 0:
            sat_level=sess.run(-loss, feed_dict=feed_dict)
            print(i, "sat level ----> ", sat_level)
            if sat_level > .99:
                break

# evaluate the truth value of a formula ....
#     print(sess.run([Forall((x,y,z),Implies(I(x,y),
#                                            Implies(P[i](y,z),P[i](x,z))))
#                     for i in range(6)],feed_dict=feed_dict))

# evaluate the truth value of P(ct,t) where ct is a central rectangle, and
# t is a set of rectangles randomly generated.

    preds = sess.run([X(t,ct) for X in P])

# plotting the value of the relation, on the centroid of t.

    fig = plt.figure(figsize=(12,8))
    jet = cm = plt.get_cmap('jet')
    cbbst = bbst[:,:2] + 0.5*bbst[:,2:]
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.scatter(cbbst[:,0], cbbst[:,1], c=preds[j][:, 0])
    plt.show()

print('done')