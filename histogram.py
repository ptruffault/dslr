import pandas as pd
import sys
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
    size = len(data['Hogwarts House'])
    for feature in data.columns[5:]:
        plt.hist([
                [data[feature][i] for i in range(0, size) if data['Hogwarts House'][i] == "Gryffindor"],
                [data[feature][i] for i in range(0, size) if data['Hogwarts House'][i] == "Ravenclaw"],
                [data[feature][i] for i in range(0, size) if data['Hogwarts House'][i] == "Hufflepuff"],
                [data[feature][i] for i in range(0, size) if data['Hogwarts House'][i] == "Slytherin"],
            ],
            alpha = 0.5,
            color = ['red', 'blue', 'yellow', 'green'],
            label=['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin'],
            histtype = 'barstacked'
        )
        plt.xlabel("notes")
        plt.ylabel("nombres")
        plt.title(feature)
        plt.legend()
        plt.show()