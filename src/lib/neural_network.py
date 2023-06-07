import numpy as np
import scipy.special  # сигмоида expit()

sigmoid = scipy.special.expit

def sigmoid_prime(x): 
    return sigmoid(x) * (1. - sigmoid(x))

class neural_network:
        
    # ініціалізувати НМ
    def __init__(self, i_count, h_count, o_count, learning_rate) :
        # кількість вузлів у вхідному, приховуваному и вихідному шарі 
        self.i_count = i_count 
        self.h_count = h_count 
        self.o_count = o_count
        # коєффіціент навчання 
        self.lr = learning_rate 
        
        # Матриці вагів
        # self.wih = (np.random.rand(self.h_count, self.i_count) - 0.5) 
        # self.who = (np.random.rand(self.o_count, self.h_count) - 0.5)

        # Матриці вагів нормальних
        self.wih = np.random.normal(0.0, self.h_count ** -0.5, self.h_count * self.i_count).reshape((self.h_count, self.i_count))
        self.who = np.random.normal(0.0, self.o_count ** -0.5, self.o_count * self.h_count).reshape((self.o_count, self.h_count))
        
        # Функція активації
        self.activation_function = scipy.special.expit
                     

    # тренування НМ на одному прикладі
    def train(self, inputs_list, targets_list):
        targets = np.array(targets_list, ndmin=2).T
        final_outputs = self.query(inputs_list)
        
        # помилки та їх розповсюдження
        output_errors = targets - final_outputs
        hidden_errors = self.who.T @ output_errors
        
        # оновити ваги між шарами 
        self.who += self.lr * output_errors * final_outputs * (1.0 - final_outputs) @ self.outputs2.T
        self.wih += self.lr * hidden_errors * self.outputs2 * (1.0 - self.outputs2) @ self.outputs1.T
        
        return self.error(targets_list, final_outputs.T[0].tolist())
    
    # опитування НМ 
    def query(self, inputs_list):
        # вихід 1-го шару
        self.outputs1 = np.array(inputs_list, ndmin=2).T
        # вихід 2-го шару
        self.outputs2 = sigmoid(self.wih @ self.outputs1)        
        # вихід 3-го шару
        self.outputs3 = sigmoid(self.who @ self.outputs2)
        return self.outputs3

    # цільова функція 
    def error(self, target_list, final_list):
        return sum([(a - b)**2 for a, b in zip(target_list, final_list)])

    
