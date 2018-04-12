import numpy as np
import random

class perceptron_MC():

    def __init__(self, data_train, data_test, lab_train, lab_test, epoch, BIAS = 1):

        self.epoch = epoch
        self.data_train = data_train
        self.data_test = data_test
        self.lab_train = lab_train
        self.lab_test = lab_test
        self.BIAS = BIAS
        self.w_vectors = {c: np.array([0 for _ in range(self.data_train.shape[1] + 1)]) for c in range(10)}

    def shuffle_data(self):
        assert self.data_train.shape[0] == len(self.lab_train)
        data_train_s = np.empty(self.data_train.shape, dtype=self.data_train.dtype)
        lab_train_s = np.empty(self.lab_train.shape, dtype=self.lab_train.dtype)
        permutation = np.random.permutation(len(self.lab_train))
        for old_index, new_index in enumerate(permutation):
            data_train_s[new_index] = self.data_train[old_index]
            lab_train_s[new_index] = self.lab_train[old_index]
        return data_train_s, lab_train_s

    def fit(self):
        update_fact = 2.
        for e in range(self.epoch):

            data_train_s, lab_train_s = self.shuffle_data()

            for x, y in zip(data_train_s, lab_train_s):

                x = np.append(x, [self.BIAS])

                arg_max, predicted_class = 0, lab_train_s[0]

                for l in range(10):
                    current_activation = np.dot(x, self.w_vectors[l])
                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, l

                if (predicted_class != y):
                    x *= update_fact
                    self.w_vectors[y] += x
                    self.w_vectors[predicted_class] -= x
            nb_err, total = self.eval_model();
            error_rate = nb_err / total
            update_fact *= .9
            print("epoch : ", e, " -> le taux d'erreur sur les données de test est de : ", error_rate)
            # print("epoch : ", e, " le taux d'erreur est de : ", print(err_rate))

    def predict(self, single_data):

        single_data = np.append(single_data, [self.BIAS])

        arg_max, predicted_class = 0, self.lab_train[0]

        for l in range(10):
            current_activation = np.dot(single_data, self.w_vectors[l])
            if current_activation >= arg_max:
                arg_max, predicted_class = current_activation, l

        return predicted_class

    def eval_model(self):
        nb_err = 0
        total = self.data_test.shape[0]
        for s_data, s_lab in zip(self.data_test, self.lab_test):
            if(self.predict(s_data) != s_lab):
                nb_err += 1
            else:
                nb_err += 0
        return nb_err, total

