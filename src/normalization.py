

class Minmax():
    def __init__(self):
        self.min = 0.
        self.max = 0.

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def apply(self, X):
        e = 1e-20
        mnmx = (X - self.min) / (self.max - self.min + e)
        return mnmx

    def unapply(self, X):
        e = 1e-20
        return (X * (self.max - self.min + e)) + self.min

    @staticmethod
    def transform(X):
        e = 1e-20
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        return (X - mn) / (mx - mn + e)


class Zscore():
    def __init__(self):
        self.mean = 0.
        self.std = 0.

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def apply(self, X):
        e = 1e-20
        return (X - self.mean) / (self.std + e)

    def unapply(self, X):
        e = 1e-20
        return (X * (self.std + e)) + self.mean

    @staticmethod
    def transform(X):
        e = 1e-20
        return (X - X.mean) / (X.std + e)
