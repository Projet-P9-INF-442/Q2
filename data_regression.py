import exemple
import numpy as np
liste_personnes=['Theresa','May' ,'Dan', 'Berrebbi',  'Leo', 'Varadkar', 'David', 'Davisand', 'Boris', 'Johnson']
# je n'apprend pas sur le dernier paragraphe !


class data:
    def __init__(self, _fichier, param, _taille_test):
        self.exemple=exemple.exemple(_fichier,param)
        self.corpus=self.exemple.corpus
        self.taille_test=_taille_test
        self.X_train=[]
        self.Y_train=[]
        self.X_test=[]
        self.Y_test=[]

    def training(self):
        self.exemple.entrainement()

    def create_data(self,liste_personnes,liste_bis):
        train, test = self.corpus[:-self.taille_test], self.corpus[-self.taille_test:]
        X, Y = [], []
        for x in train:
            for w in x:
                X.append(self.exemple.model.word_vec(w))
                if w in liste_personnes:
                    bin = 1
                else:
                    bin = 0
                Y.append(bin)

        self.X_train = np.array(X)
        self.Y_train = np.array(Y)

        X_test, Y_test = [], []
        for x in test:
            for w in x:
                X_test.append(self.exemple.model.word_vec(w))
                if w in liste_bis:
                    bin = 1
                else:
                    bin = 0
                Y_test.append(bin)
        self.X_test=np.array(X_test)
        self.Y_test=np.array(Y_test)



"""train, test = corpus_training[:3] , corpus_training[-1:]   #attention il faut enlever le dernier paragraphe pour le train
X,Y=[],[]
for x in train:
    for w in x:
        X.append(model.word_vec(w))
        if w in liste_personnes:
            bin=1
        else :
            bin=0
        Y.append(bin)

X=np.array(X)
Y=np.array(Y)

liste_personnes=['Theresa','May' ,'Dan', 'Berrebbi',  'Leo', 'Varadkar', 'David', 'Davisand', 'Boris', 'Johnson','Nicola', 'Sturgeon','Joseph','Sacuto']
X_test, Y_test = [], []
for x in test:
    for w in x:
        X_test.append(model.word_vec(w))
        if w in liste_personnes:
            bin=1
        else :
            bin=0
        Y_test.append(bin)
"""