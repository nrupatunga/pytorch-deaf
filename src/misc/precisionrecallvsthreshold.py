import math
import os

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class PrecisionRecallvsThreshold:
	def __init__(self, nbClasses, minValue=0.8, maxValue=1.0):
		self.nbClasses = nbClasses
		self.minValue = minValue
		self.maxValue = maxValue
		self.nbBins = 24
		self.histoTruePos  = np.ndarray((nbClasses, self.nbBins), dtype=np.int32)
		self.histoFalsePos = np.ndarray((nbClasses, self.nbBins), dtype=np.int32)
		self.histoFalseNeg = np.ndarray((nbClasses, self.nbBins), dtype=np.int32)
		self.histoTruePos .fill(0)
		self.histoFalsePos.fill(0)
		self.histoFalseNeg.fill(0)

		self.precision = np.ndarray((nbClasses, self.nbBins), dtype=np.float)
		self.recall    = np.ndarray((nbClasses, self.nbBins), dtype=np.float)


	def append(self, expectedClasses, resultClasses, values):
		for i in range(0, expectedClasses.size()[0]):
			binIndex = self._valueToBin( values[i] )
			if expectedClasses[i] == resultClasses[i]:
				for j in range(0, binIndex):
					self.histoTruePos[expectedClasses[i]][j] += 1
				continue
			for j in range(0, self.nbBins):
				self.histoFalseNeg[expectedClasses[i]][j] += 1
			for j in range(0, binIndex):
				self.histoFalsePos[expectedClasses[i]][j] += 1

	def computePrecisionRecall(self):
		for j in range(0, self.nbClasses):
			for i in range(0, self.nbBins):
				self.precision[j][i] = float(self.histoTruePos[j][i]) / float(self.histoTruePos[j][i] + self.histoFalsePos[j][i] + 0.00001)
				self.recall[j][i]    = float(self.histoTruePos[j][i]) / float(self.histoTruePos[j][i] + self.histoFalseNeg[j][i] + 0.00001)

	def saveGraphs(self, filename):
		self.computePrecisionRecall()

		xAxis = np.ndarray((self.nbBins), dtype=np.float)
		for i in range(0, self.nbBins):
			xAxis[i] = self._binToValue(i)

		# define the figure size and grid layout properties
		figsize = (20, 16)
		cols = 2
		gs = gridspec.GridSpec(self.nbClasses // cols + 1, cols)
		gs.update(hspace=0.5, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
		# define the data for cartesian plots
		x = np.linspace(self.minValue, self.maxValue, 16)
		y = x

		fig1 = plt.figure(num=1, figsize=figsize)
		ax = []
		offset = (self.maxValue-self.minValue) / (8 * x.size)
		for j in range(0, self.nbClasses):
		    row = (j // cols)
		    col = j % cols
		    ax.append(fig1.add_subplot(gs[row, col]))
		    ax[-1].set_title('class=%d' % j)
		    ax[-1].set_ylim([0.96, self.maxValue])
		    ax[-1].plot(xAxis, self.precision[j], 'ob', ls='-')
		    ax[-1].plot(xAxis+offset, self.recall[j], 'og', ls='-')

		# Legend
		precPatch = mpatches.Patch(color='blue', label='Precision')
		recaPatch = mpatches.Patch(color='green', label='Recall')
		fig1.legend(handles=[precPatch, recaPatch])

		plt.savefig(filename)
		plt.clf()



	def __str__(self):
		self.computePrecisionRecall()

		str = [ 'Precision:\n' ]
		str += '     '
		for i in range(0, self.nbBins):
			str += '{:>5.2f} '.format(self._binToValue(i))
		str += '\n'
		for j in range(0, self.nbClasses):
			str += '{:>2d} : '.format(j)
			for i in range(0, self.nbBins):
				str += ('{:>5.1f} '.format(self.precision[j][i]*100))
			str += '\n'

		str += '\n'
		str += [ 'Recall:\n' ]
		str += '     '
		for i in range(0, self.nbBins):
			str += '{:>5.2f} '.format(self._binToValue(i))
		str += '\n'
		for j in range(0, self.nbClasses):
			str += '{:>2d} : '.format(j)
			for i in range(0, self.nbBins):
				str += ('{:>5.1f} '.format(self.recall[j][i]*100))
			str += '\n'

		return ''.join(str)

	def _valueToBin(self, value):
		output = int(1 + self.nbBins * (value-self.minValue) / (self.maxValue-self.minValue))
		if output >= self.nbBins:
			return self.nbBins
		if output < 0:
			return 0
		return output

	def _binToValue(self, binIndex):
		return self.minValue + (self.maxValue - self.minValue) * float(binIndex) / float(self.nbBins)
