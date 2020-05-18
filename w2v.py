import numpy as np
from collections import defaultdict

# nous nous sommes inspirés pour ce code des extraits de :
# https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281
# afin de comrendre comment monter un réseau de neuronnes à la main


class word2vec():
    def __init__(self,param):
        self.n = param['n']
        self.lr = param['learning_rate']
        self.steps = param['steps']
        self.window = param['taille_contexte']


    def data_train(self, param, corpus):
        repartition = defaultdict(int)
        for ligne in corpus:
            for word in ligne:
                repartition[word] += 1

        self.taille_voc = len(repartition.keys())
        self.liste_unigram = sorted(list(repartition.keys()), reverse=False)
        self.index = dict((word, i) for i, word in enumerate(self.liste_unigram))
        self.index_word = dict((i, word) for i, word in enumerate(self.liste_unigram))

        training_data = []
        for st in corpus:
            lg = len(st)

            for i, word in enumerate(st):
                context_words = []
                target_words = self.OneHot(st[i])
                
                for k in range(i - self.window, i + self.window + 1):
                    if k != i and k <= lg - 1 and k >= 0:
                        context_words.append(self.OneHot(st[k]))
                training_data.append([target_words, context_words])
        return np.array(training_data)

    # fonction d'activation
    def softmax(self, x):
        expo = np.exp(x - np.max(x))
        return expo / expo.sum(axis=0)

    def forward(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # one hot encoding
    def OneHot(self, mot):
        vecteur = [0 for i in range(0, self.taille_voc)]
        index = self.index[mot]
        vecteur[index] = 1
        return vecteur


    def retour(self, somme, u, x):
        # etape de backpropagation
        a = np.outer(u, somme)
        b = np.outer(x, np.dot(self.w2, somme.T))
        self.w1 , self.w2 = self.w1 - (self.lr * b),  self.w2 - (self.lr * a)



    # train
    def train(self, training_data):
        # matrice 1
        self.w1 = np.random.uniform(-1., 1., (self.taille_voc, self.n))
        # matrice 2
        self.w2 = np.random.uniform(-1., 1., (self.n, self.taille_voc))

        for i in range(0, self.steps):
            self.perte = 0

            for target_word, context_word in training_data:
                y_pred, h, u = self.forward(target_word)

                somme = np.sum([np.subtract(y_pred, word) for word in context_word], axis=0)
                self.retour(somme, h, target_word)
                self.perte += -np.sum([u[word.index(1)] for word in context_word]) + len(context_word) * np.log(np.sum(np.exp(u)))
                # self.perte += -2*np.log(len(context_word)) -np.sum([u[word.index(1)] for word in context_word]) + (len(context_word) * np.log(np.sum(np.exp(u))))
            #if i%5==0:
            print('step:', i, 'perte:', self.perte)
