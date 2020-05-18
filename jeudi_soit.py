from sklearn.externals import joblib
donne=joblib.load("oliver_twist_dim300.pkl")

l=[]
for x in donne.corpus:
    for k in x:
        l.append(k)

vect=[]
for x in l:
    vect.append(donne.exemple.model.word_vec(x))

from sklearn.cluster import KMeans

clust=KMeans(n_clusters=10,  init='k-means++', n_init=10, max_iter=300,verbose=1)

res=clust.fit(vect)

v=donne.exemple.model.word_vec("Oliver").reshape(-1,1).T
print(res.predict(v))

import numpy as np
labels = res.labels_
np.histogram(labels, bins=10)

joblib.dump(res, "clusters.pkl")

cluster=[]
ponctuation_et_stop_word=['.',',',';',':','!','?','The','He','At','She','I','They','His','This','As','That','There','When']

v = donne.exemple.model.word_vec("Oliver").reshape(-1, 1).T
nb=res.predict(v)

for mot in l:
    v = donne.exemple.model.word_vec(mot).reshape(-1, 1).T
    if res.predict(v)==nb and mot not in ponctuation_et_stop_word and mot.istitle()==True:
        cluster.append(mot)

cluster_sans_doublon=[]
for mot in cluster:
    if mot not in cluster_sans_doublon:
        cluster_sans_doublon.append(mot)

noms_pred=cluster_sans_doublon


####################################################
####################################################

import re

def anonymisation(str,liste_noms):
    texte=re.split("[ .,;?!:']", str)
    for k in range(len(texte)):
        if texte[k] in liste_noms:
            texte[k]='ANONYME'
    return ' '.join(texte)

anonymisation(oliver,noms_pred)

text_file = open("sample.txt", "wt")
n = text_file.write(anonymisation(oliver,noms_pred))
text_file.close()

# on regarde maintenant les métriques avec les fonctions du fichier REGEXP

def maj_to_class(str,liste):
    labels=[]
    for x in re.split('[ .,;?!:]',str):
        if x in liste:
            labels.append((x,1))
        else :
            labels.append((x,0))
    return labels

pred = maj_to_class(oliver,noms_pred)
true = true_label(oliver,liste_noms)
conf=confusion(pred,true)

res=metrique(conf)