#   Names: Robert Morabito 7093230
#          Tyler McDonald 7235542
#   Date: 2024-04-21
#   Title: COSC 4P82 - Final Project

import time
import random
from multiprocessing import Pool, cpu_count
from deap import base
from GP import WordPredictGP
from iofunction import parse_csv, parse_json, write_to_csv, write_qual
from vectorizer import train_word2vec, w2v_vectorize, bert_vectorize, unvectorize


def main():
    # Initialize random seed using current seconds
    seed = time.time()
    random.seed(seed)

    # Import training and testing data and parameters, and randomizing datasets
    train = parse_csv('data/MNH-Training-Scaled.csv')
    random.shuffle(train)
    test = parse_csv('data/MNH-Testing-Scaled.csv')
    random.shuffle(test)
    params = parse_json('data/parameters.json')

    # Vectorize datasets using word2vec
    train_word2vec(train + test)
    train_vec = w2v_vectorize(train)
    test_vec = w2v_vectorize(test)
    # train_vec = bert_vectorize(train)
    # test_vec = bert_vectorize(test)

    # Initialize multiprocessing pool
    pool = Pool(cpu_count())
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)

    # Run GP evolution for training and testing
    gp = WordPredictGP(train_vec, test_vec, params, toolbox=toolbox)  # Pass toolbox as an argument
    logs, predicts, test_fit = gp.run_gp()

    # Close the pool and wait for the work to complete
    pool.close()
    pool.join()

    # Write results
    write_to_csv('data/Results.csv', logs, seed, test_fit)

    # Write qualitative results
    begins = []
    targets = []
    predicts = unvectorize(predicts)
    for i in range(len(predicts)):
        begins.append(test[i][:-1])
        targets.append(test[i][-1])
    write_qual('data/Qualitative.txt', begins, targets, predicts)


if __name__ == '__main__':
    main()
