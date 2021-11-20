from tools.logreg import LogReg
from tools.Features import Features
import sys
import csv
import os
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser("See histogram from csv file")
    args.add_argument("file", help="dataset filepath")
    args = args.parse_args()
    data = Features(args.file)
    models = []
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    for house in houses:
        filename = './models/' + house.lower() + '-model.csv'
        LR = LogReg(data.features)
        LR.load(filename)
        models.append(LR.model())
    results = [None] * LR.size
    for predictions, house in zip(models, houses):
        for prediction, i in zip(predictions, range(0, LR.size)):
            if results[i] is None or prediction > results[i]['proba']:
                results[i] = {'house': house, 'proba': prediction}
    with open('houses.csv', 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Hogwarts House'])
        for result, i in zip(results, range(0, LR.size)):
            writer.writerow([i, result['house']])