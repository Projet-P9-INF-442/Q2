import w2v
import corpus
import corpus_bigram


param = {}
param['n'] = 300                # dimension of word embeddings
param['taille_contexte'] = 4        # context window +/- center word
param['min_count'] = 0           # minimum word count
param['epochs'] = 800           # number of training epochs
param['neg_samp'] = 100           # number of negative words to use during training
param['learning_rate'] = 0.01    # learning rate
#np.random.seed(0)                   # set the seed for reproducibility


class exemple:
    def __init__(self,_fichier,param):
        self.corpus = corpus.corpus(_fichier).generer()
        self.model = w2v.word2vec(param)

    def entrainement(self):
        training_data = self.model.data_train(param, self.corpus)
        self.model.train(training_data)


"""#corp=corpus.corpus("texte.txt")
#corpus_training=corp.generer()

# INITIALIZE W2V MODEL
model = w2v.word2vec(param)

# generate training data
training_data = model.data_train(param, corpus_training)

import time
t1=time.time()

# train word2vec model
model.train(training_data)

print(time.time()-t1, "secondes.")"""

"""# je save le modele
from sklearn.externals import joblib
joblib_file = "model_brexit.pkl"
joblib.dump(model, joblib_file)

# pr le reload apres :
mod = joblib.load(joblib_file)
"""