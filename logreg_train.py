from tools.Features import Features
from tools.logreg import LogReg
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser("See histogram from csv file")
    args.add_argument("file", help="dataset filepath")
    args = args.parse_args()
    data = Features(args.file)
    train = {}
    houses_filters = {
        'Ravenclaw': [
            #"Birthday",
            #"Best Hand",
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
            #"Birthday",
            #"Best Hand",
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
            #"Birthday",
            #"Best Hand",
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
            #"Birthday",
            #"Best Hand",
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

    iteration = 250
    learning_rate = 0.1
    for house in set(data.houses):
        Y = [1 if house == h else 0 for h in data.houses]
        X = data.filter(houses_filters[house])
        LR = LogReg(X, Y, iteration, learning_rate)
        LR.train()
        LR.save('./models/'+house.lower()+'-model.csv')
        err = LR.accuracy()
        print(house, err, 100 - err / LR.size * 100)
