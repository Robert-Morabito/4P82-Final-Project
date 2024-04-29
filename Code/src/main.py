#   Names: Robert Morabito 7093230
#          Tyler McDonald 7235542
#   Date: 2024-04-21
#   Title: COSC 4P82 - Final Project

import time
import random
from GP import WordPredictGP
from iofunction import parse_csv, parse_json
from vectorizer import train_word2vec, vectorize, unvectorize


def main():
    # Initialize random seed using current seconds
    #random.seed(int(time.time()))
    random.seed(10)  # Temp for testing

    # Import training and testing data and parameters, and randomizing datasets
    train = parse_csv('data/MNH-Training-Scaled.csv')
    random.shuffle(train)
    test = parse_csv('data/MNH-Testing-Scaled.csv')
    random.shuffle(test)
    params = parse_json('data/parameters.json')

    # Vectorize datasets using word2vec
    train_word2vec(train+test)
    train_vec = vectorize(train)
    test_vec = vectorize(test)

    # Run GP evolution for training
    gp = WordPredictGP(train_vec, test_vec, params)
    gp.train_gp()

    # Run GP testing

    # Return vectorized testing results back to english and save out to a .csv


if __name__ == '__main__':
    main()
