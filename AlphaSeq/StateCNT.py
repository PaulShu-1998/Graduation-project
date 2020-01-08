import numpy as np


class Dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class HarshState():
    def __init__(self, overallSteps):
        """

        :param overallSteps: the number of all elements in matrix
        """
        self.overallSteps = overallSteps
        self.temp = np.power(2, np.arange(self.overallSteps), dtype=np.int64)  # 是一个2^0 ... 2^overallSteps 的array
        self.visitedState = {0: 0}
        self.visitedState1 = {0: 0}

    def store(self, cummulativeMove):
        cummulativeMove = (cummulativeMove + 1) / 2  # turn -1 to 0
        state = np.append(cummulativeMove, np.zeros(self.overallSteps - len(cummulativeMove)))  # 补全未填充的部分
        # state to integer
        stateInt = np.dot(state, self.temp).astype(np.int64)
        # store
        self.visitedState[stateInt] = 0
        self.visitedState1[stateInt] = 0

    def printCnt(self):
        return len(self.visitedState)

    def renew(self):
        self.visitedState1 = {0: 0}

    def printCnt1(self):
        return len(self.visitedState1)
