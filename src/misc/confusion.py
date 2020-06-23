import math

import numpy as np


class ConfusionMatrix:
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMatrix constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems.

    Args:
        k (int): number of classes in the classification problem

    """

    def __init__(self, k, classNames = None, summary = False):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.k = k
        self.classNames = classNames

        self.valids = np.ndarray((k), dtype=np.float)
        self.unionvalids = np.ndarray((k), dtype=np.float)
        self.totalValid = 0
        self.averageValid = 0
        self.summary = summary

        self.reset()

    def reset(self):
        self.conf.fill(0)
        self.valids.fill(0)
        self.unionvalids.fill(0)
        self.totalValid = 0
        self.averageValid = 0

    def test(self):
        self.conf     = np.array([ [123, 4, 3, 2, 3], 
                           [11, 82, 10, 0, 0], 
                           [4, 3, 81, 0, 0], 
                           [2, 0, 0, 6, 0], 
                           [8, 0, 0, 0, 6] ])
        print(self)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        return self.conf

    def updateValids(self):
        total = 0
        for t in range(self.k):
            self.valids[t] = self.conf[t, t] / float(self.conf[t,:].sum())
            self.unionvalids[t] = self.conf[t, t] / float(self.conf[:,t].sum()+self.conf[t,:].sum()-self.conf[t, t])
            total = total + self.conf[t, t]

        self.totalValid = total / float(self.conf.sum())
        self.averageValid = 0
        self.averageUnionValid = 0
        nvalids = 0
        nunionvalids = 0

        for t in range(self.k):
            if not math.isnan(self.valids[t]):
                self.averageValid = self.averageValid + self.valids[t]
            nvalids = nvalids + 1

            if not math.isnan(self.valids[t]) and not math.isnan(self.unionvalids[t]):
                self.averageUnionValid = self.averageUnionValid + self.unionvalids[t]
            nunionvalids = nunionvalids + 1

        self.averageValid = self.averageValid / float(nvalids)
        self.averageUnionValid = self.averageUnionValid / float(nunionvalids)

    def fullStr(self):
        maxCnt = self.conf.max()
        nDigits = 1 + math.ceil( math.log10(maxCnt))
        if nDigits < 8:
            nDigits = 8
        format = '%%%dd' % nDigits

        str = [ 'ConfusionMatrix:\n' ]

        for t in range(self.k):
            name = ""
            if self.classNames is not None:
                name = "[ %s ]" % self.classNames[t]
            str += ' ['
            for p in range(self.k):
                str += format % self.conf[t][p]
            str += ']  %03.3f%% \t%s\n' % (self.valids[t] * 100, name)

        str += ' + average row correct: %2.3f%%\n' % (self.averageValid*100)
        str += ' + average rowUcol correct (VOC measure): %2.3f%%\n' % (self.averageUnionValid*100)
        str += ' + global correct: %2.3f%%\n' % (self.totalValid*100)
        return ''.join(str)

    def summaryStr(self):
        maxCnt = self.conf.max()
        nDigits = 1 + math.ceil( math.log10(maxCnt))
        if nDigits < 8:
            nDigits = 8
        format = '%%%dd' % nDigits

        str = [ 'ConfusionMatrix summary:\n' ]
        str += ' + average row correct: %2.3f%%\n' % (self.averageValid*100)
        str += ' + average rowUcol correct (VOC measure): %2.3f%%\n' % (self.averageUnionValid*100)
        str += ' + global correct: %2.3f%%\n' % (self.totalValid*100)
        return ''.join(str)

    def __str__(self):
        self.updateValids()
        if self.summary:
            return self.summaryStr()
        return self.fullStr()
