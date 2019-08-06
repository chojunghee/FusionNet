from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
#from .optimizer import Optimizer

class _scheduler_learning_rate(object):
    
	def __init__(self, optimizer, epoch=-1):

		self.optimizer 	= optimizer
		self.epoch 		= epoch	

	def step(self, epoch=None):
	
		if epoch is None:
            
			epoch = self.epoch + 1
		
		self.epoch	= epoch
		lr 			= self.get_lr()

		for param_group in self.optimizer.param_groups:
        
			param_group['lr'] = lr

class scheduler_learning_rate_sigmoid(_scheduler_learning_rate): 

	def __init__(self, optimizer, lr_initial, lr_final, numberEpoch, alpha=10, beta=0, epoch=-1):

		_index 		= np.linspace(-1, 1, numberEpoch)
		_sigmoid	= 1 / (1 + np.exp(alpha * _index + beta))

		val_initial = _sigmoid[0]
		val_final	= _sigmoid[-1]

		a = (lr_initial - lr_final) / (val_initial - val_final)
		b = lr_initial - a * val_initial 

		self.schedule		= a * _sigmoid + b
		self.numberEpoch	= numberEpoch

		super(scheduler_learning_rate_sigmoid, self).__init__(optimizer, epoch)

	def get_lr(self, epoch=None):

		if epoch is None:

			epoch = self.epoch

		lr = self.schedule[epoch]

		return lr

	def plot(self):

		fig 	= plt.figure()
		ax		= fig.add_subplot(111)	
		
		ax.plot(self.schedule)
		
		plt.xlim(0, self.numberEpoch + 1)
		plt.xlabel('epoch')
		#plt.ylabel('learning rate')
		plt.grid(linestyle='dotted')
		plt.tight_layout()
		plt.show()

class scheduler_learning_rate_logistic(_scheduler_learning_rate): 

	def __init__(self, optimizer, lr_initial, lr_final, numberEpoch, mu=0, sigma=0.25, epoch=-1):

		_index 		= np.linspace(-10, 10, numberEpoch)
		_logistic	= 0.5 + 0.5 * np.tanh(-(_index - mu) / (2*sigma))

		val_initial = _logistic[0]
		val_final	= _logistic[-1]

		a = (lr_initial - lr_final) / (val_initial - val_final)
		b = lr_initial - a * val_initial 

		self.schedule		= a * _logistic + b
		self.numberEpoch	= numberEpoch

		super(scheduler_learning_rate_logistic, self).__init__(optimizer, epoch)

	def get_lr(self, epoch=None):

		if epoch is None:

			epoch = self.epoch

		lr = self.schedule[epoch]

		return lr

	def plot(self):

		fig 	= plt.figure()
		ax		= fig.add_subplot(111)	
		
		ax.plot(self.schedule)
		
		plt.xlim(0, self.numberEpoch + 1)
		plt.xlabel('epoch')
		#plt.ylabel('learning rate')
		plt.grid(linestyle='dotted')
		plt.tight_layout()
		plt.show()


class scheduler_learning_rate_logistic2(_scheduler_learning_rate): 

	def __init__(self, optimizer, lr_initial, lr_mid, lr_final, numberEpoch, mid_start, mid_end, mu=0, sigma=0.25, epoch=-1):

		_index_1 		= np.linspace(-1, 1, mid_start)
		_index_2 		= np.linspace(-1, 1, numberEpoch - mid_end)
		""" _logistic_1		= 0.5 + 0.5 * np.tanh((_index_1 - mu) / (2*sigma))
		_logistic_2		= 0.5 + 0.5 * np.tanh(-(_index_2 - mu) / (2*sigma))
 """
		_sigmoid_1	= 1 / (1 + np.exp(10 * _index_1 ))
		_sigmoid_2	= 1 / (1 + np.exp(10 * _index_2 ))

		val_initial_1 = _sigmoid_1[0]
		val_initial_2 = _sigmoid_2[0]

		val_final_1   = _sigmoid_1[-1]
		val_final_2   = _sigmoid_2[-1]

		a1 = (lr_initial - lr_mid) / (val_initial_1 - val_final_1)
		a2 = (lr_final - lr_mid) / (val_final_2 - val_initial_2)

		b1 = lr_initial - a1 * val_initial_1 
		b2 = lr_mid 	- a2 * val_initial_2 

		schedule_1		= a1 * _sigmoid_1 + b1
		schedule_2		= a2 * _sigmoid_2 + b2

		self.schedule   = lr_mid * np.ones(numberEpoch)
		self.schedule[:mid_start] = schedule_1
		self.schedule[mid_end:]   = schedule_2
		self.numberEpoch	= numberEpoch

		super(scheduler_learning_rate_logistic2, self).__init__(optimizer, epoch)

	def get_lr(self, epoch=None):

		if epoch is None:

			epoch = self.epoch

		lr = self.schedule[epoch]

		return lr

	def plot(self):

		fig 	= plt.figure()
		ax		= fig.add_subplot(111)	
		
		ax.plot(self.schedule)
		
		plt.xlim(0, self.numberEpoch + 1)
		plt.xlabel('epoch')
		#plt.ylabel('learning rate')
		plt.grid(linestyle='dotted')
		plt.tight_layout()
		plt.show()