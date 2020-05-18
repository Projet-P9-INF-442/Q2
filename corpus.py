from nltk.tokenize import TweetTokenizer

class corpus():
    def __init__(self,_fichier):
        self.fichier = _fichier

    def generer(self):
        tknzr = TweetTokenizer()
        rep = []
        file = open(self.fichier, "r")
        lines = file.readlines()
        file.close()
        for l in lines:
            lg = tknzr.tokenize(l)
            if len(lg) > 0:
                rep.append(lg)
        return rep



