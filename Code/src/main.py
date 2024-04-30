#   Names: Robert Morabito 7093230
#          Tyler McDonald 7235542
#   Date: 2024-04-21
#   Title: COSC 4P82 - Final Project

import time
import os
import random
from deap import base
from GP import WordPredictGP
from iofunction import parse_csv, parse_json, write_to_csv, write_qual
from vectorizer import train_word2vec, w2v_vectorize, unvectorize


def main():
    dirname = os.path.dirname(__file__)
    

    # Import training and testing data and parameters, and randomizing datasets
    train = parse_csv('data/MNH-Training-Scaled.csv')
    random.shuffle(train)
    test = parse_csv('data/MNH-Testing-Scaled.csv')
    random.shuffle(test)

    # Vectorize datasets using word2vec
    train_word2vec(train + test)
    train_vec = w2v_vectorize(train)
    test_vec = w2v_vectorize(test)
    # train_vec = bert_vectorize(train)
    # test_vec = bert_vectorize(test)

    for paramset in range(0,5):
        print(f'param set {paramset}')
        # Initialize multiprocessing pool
        toolbox = base.Toolbox()

        for paramrun in range(0,5):
            # Initialize random seed using current seconds
            seed = time.time()
            random.seed(seed)
            print(f'param set {paramset} iter {paramrun}')
            # Run GP evolution for training and testing
            gp = WordPredictGP(train_vec, test_vec, parse_json(f'data/parameters{paramset}.json'), toolbox=toolbox)  # Pass toolbox as an argument
            logs, predicts, test_fit, best_tree = gp.run_gp()


            # Write results
            write_to_csv(f'Results-{paramset}{paramrun}.csv', logs, seed, test_fit, best_tree)

            # Write qualitative results
            begins = []
            targets = []
            predicts = unvectorize(predicts)
            for i in range(len(predicts)):
                begins.append(test[i][:-1])
                targets.append(test[i][-1])
            write_qual(f'Qualitative-{paramset}{paramrun}.txt', begins, targets, predicts)


if __name__ == '__main__':
    main()
