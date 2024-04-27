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


def parse_json(filename):
    """
    This function simply parses a json file given a file path
    :param filename: Name of .json file
    :return: Parsed json file
    """
    with open(filename, 'r') as p:
        params = json.load(p)
    return params
