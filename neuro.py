import numpy as np
import scipy.special as sci


class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# количество входных, скрытых, выходных узлов
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes
		# коэффициент обучения
		self.lr = learning_rate
		# матрицы весов
		# self.wih = (np.random.rand(self.hnodes, self.inodes)- 0.5)
		# self.who = (np.random.rand(self.onodes, self.hnodes)- 0.5)
		# self.wih = (np.random.rand(self.hnodes, self.inodes)*2.0 - 1.0)
		# self.who = (np.random.rand(self.onodes, self.hnodes)*2.0 - 1.0)
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		pass

	def activation_function(self, x):
		return sci.expit(x) # сигмоида

	def train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
		pass

	def query(self, inputs_list):
		# преобразовать список входных значений
		# в двухмерный массив
		inputs = np.array(inputs_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		return final_outputs