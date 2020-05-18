from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.externals import joblib
import data_regression

param = {}
param['n'] = 300                # dimension   of word embeddings
param['taille_contexte'] = 4        # context window +/- center word
param['min_count'] = 0           # minimum word count
param['steps'] = 150           # number of training steps
param['neg_samp'] = 100           # number of negative words to use during training
param['learning_rate'] = 0.01

donne=data_regression.data("texte.txt",param,1)
donne.training()

from sklearn.externals import joblib
joblib.dump(donne, "oliver_twist_dim300.pkl")

donne=joblib.load("oliver_twist_dim300.pkl")

liste_personnes=['Oliver',"Bumble",'Twist','Charley'] #liste generer par le fichier spacy_nom_propre_pour_train
liste_bis=liste
donne.create_data(liste_personnes,liste_bis)
X,Y,X_test,Y_test=donne.X_train,donne.Y_train,donne.X_test,donne.Y_test

clf = LogisticRegression(random_state=0).fit(X, Y)
print("score : ")
print("train : ", clf.score(X,Y))
print("test", clf.score(X_test,Y_test))

#on save la regression
#from sklearn.externals import joblib
#joblib_file = "data_vect_dim_100_gd_texte_ca_marche_voir_verif.pkl"
#joblib.dump(donne, joblib_file)


"""vect1=np.array([list(donne.exemple.model.word_vec("Theresa"))])
vect2=np.array([list(donne.exemple.model.word_vec("Berrebbi"))])
vect3=np.array([list(donne.exemple.model.word_vec("Varadkar"))])
vect4=np.array([list(donne.exemple.model.word_vec("Sturgeon"))])
vect5=np.array([list(donne.exemple.model.word_vec("Davisand"))])
vect6=np.array([list(donne.exemple.model.word_vec("Donald"))])

print(clf.predict_proba(vect1))
print(clf.predict_proba(vect2))
print(clf.predict_proba(vect3))
print(clf.predict_proba(vect4))
print(clf.predict_proba(vect5))
print(clf.predict_proba(vect6))"""

def proba(nom):
    vect1=np.array([list(donne.exemple.model.word_vec(nom))])
    return clf.predict_proba(vect1)

def vect(nom):
    return np.array([list(donne.exemple.model.word_vec(nom))])

def sim(v,w):   #la similarité c pas pertinent, seuls certaines features traduisent du fait que c un nom propre ou non
    theta_num = np.dot(v,w.T)
    theta_den = np.linalg.norm(v) * np.linalg.norm(w)
    return  theta_num / theta_den