from mlib.Datasets import Datasets
from mlib.MultiLayerPerceptron import MultipleLayerPerceptron as mlp

if __name__ == '__main__':
    features = [
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
    dataset = Datasets('./datasets/dataset_train.csv', "Hogwarts House", features)
    mlp = mlp(
        dataset=dataset,
        structure=[8],
        #loss="MeanSquaredError"
    )
    mlp.train(
        iterations=1000,
        learning_rate=0.001
    )
    mlp.save('model.json', dataset)
    print('End accuracy :', mlp.accuracy, '%')

