import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    plt.scatter(data['Defense Against the Dark Arts'], data['Astronomy'])
    plt.xlabel("Defense Against the Dark Arts")
    plt.ylabel("'Astronomy")
    plt.show()