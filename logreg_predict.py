from tools.logreg import LogReg
from tools.Features import Features
import sys
import csv
import os
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("file", help="dataset filepath")
    args = args.parse_args()
    data = Features(args.file)
    models = []
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    houses_filters = {
        'Ravenclaw': [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying"
        ],
        'Slytherin': [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying"
        ],
        'Gryffindor': [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying"
        ],
        'Hufflepuff': [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying"
        ]
    }
    for house in houses:
        filename = './models/' + house.lower() + '-model.csv'
        X = data.filter(houses_filters[house])
        LR = LogReg(X)
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