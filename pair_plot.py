import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

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
    for features in data.columns:
        if i != 1 and i < 6:
            del data[features]
        i += 1
    sns.pairplot(data, hue='Hogwarts House', height=1.5)
    plt.show()