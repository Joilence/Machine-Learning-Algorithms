from DecisionTree import DecisionTree
import pandas as pd
import numpy as np
import threading
import multiprocessing


class RandomForest:

    def __init__(self, max_depth=10, min_size=1, sample_fraction=0.7, n_trees=10, feature_fraction=1.0):
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_fraction = sample_fraction
        self.n_trees = n_trees
        self.feature_fraction = feature_fraction

    def fit(self, X, y, paral=-1):
        self.trees = list()
        dataset = X.tolist()

        for i in range(len(dataset)):
            index = len(dataset[0])
            dataset[i].append(y[i])

        if paral > 1:
            pool = multiprocessing.Pool(paral)
            
        else:
            for i in range(self.n_trees):
                sample = self.sampling(dataset, self.sample_fraction)
                # tree = DecisionTree(max_depth=self.max_depth,
                #                     min_size=self.min_size)
                # tree.fit(sample)
                tree = single_tree_fit(self.max_depth, self.min_size, sample)
                self.trees.append(tree)

    def single_tree_fit(self, max_depth, min_size, sample):
        tree = DecisionTree(max_depth, min_size)
        tree.fit(sample)
        return tree

    def bagging_tree_predict(self, row):
        preds = [self.single_tree_predict(tree, row) for tree in self.trees]
        return max(set(preds), key=preds.count)

    def single_tree_predict(self, tree, row):
        pred = tree.predict(row)
        return pred[0]

    def predict(self, X, paral=-1):
        pred_set = [tree.predict(X) for tree in self.trees]
        pred_set = np.matrix(pred_set).T.tolist()
        
        if paral > 1:
            pool = multiprocessing.Pool(paral)
            pred = pool.map(vote_pred, pred_set)
        else:
            pred = [self.vote_pred(preds) for preds in pred_set]

        return pred, pred_set

    def vote_pred(self, preds):
        return max(set(preds), key=preds.count)

    def sampling(self, dataset, ratio):
        sample_num = round(len(dataset) * ratio)
        sample = pd.DataFrame(dataset).sample(n=sample_num).values
        return sample
