# Put here the functions that might be required by more than 1 model/table

class pipeline:

    def __init__(self, stages):
        self.stages = stages
        self.n_stages = len(stages)

    def fit(self, X):

        for stage in self.stages:
            stage.fit(X)

    def predict(self, test):
        
        for stage in self.stages[:-1]:
            test = stage.predict(test)

        return self.stages[-1].predict(test)

    def evaluate():
        pass

    # [preprocessor, model]