import sys
import pandas as pd
from tools.Stats import *
import os
import random
from datetime import datetime

class Features(object):
    def __init__(self, path):
        if not os.path.isfile(path):
            sys.stderr.write("Ce fichier n'existe pas")
            exit(1)
        try:
            self.data = pd.read_csv(path)
        except:
            sys.stderr.write("Ce fichier n'est pas correctement formaté.")
            exit(1)
        self.houses = self.data['Hogwarts House']


    def complete_missing_value(self, features):
        for i in range(0, mylen(features)):
            for j in range(0, mylen(features[i])):
                if not is_numeric(features[i][j]):
                    data = [features[i][k] if self.houses[j] == self.houses[k] else None for k in range(0, mylen(features[i]))]
                    features[i][j] = mymean(data) + mystd(data) * random.randint(-1, 1)
        return features

    def filter(self, collumns):
        ret = []
        for feature in collumns:
            if feature in self.data:
                ret.append(self.data[feature].to_numpy())
        return self.complete_missing_value(ret)



