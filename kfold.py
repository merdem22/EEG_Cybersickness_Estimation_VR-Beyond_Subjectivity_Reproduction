from sklearn.model_selection._split import KFold, _num_samples, check_random_state
import numpy as np


class FewShotKFold(KFold):
    def __init__(self, target_class, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.target_class = target_class

    def _iter_test_indices(self, X, y, groups=None):
        samples = _num_samples(X[y == self.target_class])
        indices = np.where(y == self.target_class)[0]

        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, samples // n_splits, dtype=int)
        fold_sizes[: samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield np.concatenate([indices[:start],
                                  indices[stop:]])
            current = stop


if __name__ == '__main__':
    X = np.random.rand(20, 50)
    y = (np.random.rand(20) < 0.8).astype(int)
    print("num of ones", y.sum(), y)
    kfold = FewShotKFold(target_class=1, n_splits=5)
    for train, test in kfold.split(X, y):
        print(y[train], y[test])
