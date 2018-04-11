import pickle
import numpy as np
import random
import kMeans
import perceptronMC

##############################################
#   Traitement et préparation des données    #
##############################################

# Chargement du fichier de données à l'aide de pickle.
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

# "full_dataset" contient le fichier brut, il est traité par la suite.
full_dataset = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')
label_names = unpickle('./cifar-10-python/cifar-10-batches-py/batches.meta')[b'label_names']

# Split des données en données de training, et données de validation.
# De même pour les labels
data = full_dataset[b'data']
labels = np.asarray(full_dataset[b'labels'])
label_correspondance = {}
for l in labels:
	label_correspondance[l] = label_names[l]
for l in labels:
	l = label_correspondance[l]
labs = np.array([label_correspondance[l] for l in labels])
data_train = data[:9000]
labels_train = labels[:9000]
data_test = data[9000:]
labels_test = labels[9000:]

print(labels)
print(labs)
print(type(labels))
print(type(labs))
print(labels.shape)
print(labs.shape)

name_img = full_dataset[b'filenames']

# Dictionnaire permettant de voir quel numéro de label est associé à quelle catégorie d'image.

# Initialisaiton d'un tableau "repartition" permettant de voir le nombre d'exemple par classe ici dans le batch 1 du jeu de données.
repartition = [0,0,0,0,0,0,0,0,0,0]
for i in labels:
 	repartition[i] += 1
# for i in range(10):
#	repartition[i] = repartition[i] / len(labels)

# print(data)
# print(labels)
# print(label_names)
# print(name_img)
# print(label_correspondance)
print(repartition)

##############################################
# Approche Naïve avec k-Means et Perceptron  #
##############################################

# k-Means
# nbrClusters = 10
# epoch_for_k_means = 20

# modele_k_means = kMeans.K_Means(nbrClusters, epoch_for_k_means)
# print("Entrainement du modèle k-Means...")
# modele_k_means.fit(data)
# print("Evaluation du modèle de k-Means par clusters : ")
# modele_k_means.eval_k_means(labels)

# # Perceptron
# epoch_for_perceptron = 300

# modele_perceptron = perceptronMC.perceptron_MC(data_train, data_train, labels_train, labels_train, epoch_for_perceptron)
# print("entrainement du modèle perceptron + evolution du taux d'erreur à chaque epoch...")
# modele_perceptron.fit()
# print("fin de l'entrainement du perceptron")

###############################################
#               Approche avancée              #
###############################################

# --------------------------------------------------------------------------------------------------------------

# Il y a 10000 images représentées dans les différents batch (ici le batch 1)
# Pour ces 10000 images il y a 10 classes différentes pour la classification
# On peut facilement regarder la répartition des exemples par classe, il y a environ 10% pour chacune des classes, la distribution est relativement homogène.
# Voir "repartition" contenant précisémment la repartition des images par classes.
# Dans Cifar-10 les classes sont assimilées au catégories suivantes : automobile, bird, cat, deer, dog, frog, horse, ship, truck.
# On peut noter que la tache de classification pour un humain est relativement facile. Cependant pour une machine cela reste beaucoup plus complexe.
# Pourquoi pas réaliser un approche naïve avec un algorithm tel que K-Means, et ainsi analyser les résultats obtenus.