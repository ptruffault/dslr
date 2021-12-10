from tools.Stats import *
import pandas as pd
import os
import sys
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser("See histogram from csv file")
    args.add_argument("file", help="dataset filepath")
    args = args.parse_args()
    if not os.path.isfile(args.file):
        sys.stderr.write("Ce fichier n'existe pas")
        exit(1)
    try:
        data = pd.read_csv(args.file)
    except:
        sys.stderr.write("Ce fichier n'est pas correctement formaté.")
        exit(1)
    if 'Hogwarts House' not in data:
        sys.stderr.write("Ce fichier est mal formaté pour ce projet.")
        exit(1)
    i = 0
    data_describe = {}
    for feature in data.columns:
        if i > 6:
            data_describe[feature] = {
                'Count':    mylen(data[feature]),
                'Mean':     mymean(data[feature]),
                'Std':      mystd(data[feature]),
                'Min':      mymin(data[feature]),
                '25%':      percent(data[feature], 25),
                '50%':      percent(data[feature], 50),
                '75%':      percent(data[feature], 75),
                'Max':      mymax(data[feature]),
             
            }
        i += 1
    print(pd.DataFrame(data_describe))