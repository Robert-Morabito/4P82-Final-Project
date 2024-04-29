import csv

import pandas as pd
import json


def parse_csv(filename):
    """
    This function reads a csv file of sentences and returns them parsed into a 2D array of words
    :param filename: Name of the .CSV file
    :return: 2D array of words
    """
    # Local variables
    lines = []

    # Load data from .CSV file
    data = pd.read_csv(filename, sep=',', header=None)

    # Parse lines of data into individual words
    for index, row in data.iterrows():
        words = str(row[0]).split()
        lines.append(words)

    return lines


def write_to_csv(filename, data, seed, test):
    """
    Write GP statistics to a CSV file
    :param test: Testing scores
    :param seed: Random seed
    :param filename: Filepath to output file
    :param data: Statistics to write
    """
    with open(filename, 'w', newline='') as file:
        # Fields
        headers = ['gen', 'seed', 'avg_fit', 'max_fit', 'avg_size', 'max_size', 'test']
        write = csv.DictWriter(file, fieldnames=headers)
        write.writeheader()

        # Get statistics for writing
        gens = data.select("gen")
        avg_fit = data.chapters['fitness'].select("avg")
        max_fit = data.chapters['fitness'].select("max")
        avg_size = data.chapters['size'].select("avg")
        max_size = data.chapters['size'].select("max")

        # Write the statistics out to the csv
        for i in range(len(gens)):
            write.writerow({
                'seed': seed,
                'gen': gens[i],
                'avg_fit': avg_fit[i],
                'max_fit': max_fit[i],
                'avg_size': avg_size[i],
                'max_size': max_size[i],
            })

        write.writerow({'test': test})


def write_qual(filename, begin, tar, pred):
    """
    Writes qualitative text prediction examples to a text file.
    :param filename: File path for output
    :param begin: List of beginnings of sentences
    :param tar: List of target words for the end
    :param pred: List of predicted words for the end
    :return: None
    """
    with open(filename, 'w') as file:
        for words, target, predicted in zip(begin, tar, pred):
            sentence = ' '.join(words)
            file.write(f"Input: {sentence}. Target: {target}. Predicted: {predicted}\n")


def parse_json(filename):
    """
    This function simply parses a json file given a file path
    :param filename: Name of .json file
    :return: Parsed json file
    """
    with open(filename, 'r') as p:
        params = json.load(p)
    return params
