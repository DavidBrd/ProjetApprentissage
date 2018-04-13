import pickle
import numpy as np
import random
import kMeans
import perceptronMC

# Chargement du fichier de données à l'aide de pickle.
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

full_dataset  = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')
full_dataset2 = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_2')

data  				= np.array(full_dataset[b'data'], dtype="float64")
data2 				= np.array(full_dataset2[b'data'], dtype="float64")


labels  			= np.asarray(full_dataset[b'labels'])
labels2 			= np.asarray(full_dataset2[b'labels'])

data   				= np.append(data, data2, axis = 0)
labels 				= np.append(labels, labels2)

data_train   	= data[:5000]
labels_train 	= labels[:5000]

data_test   	= data[17000:]
labels_test 	= labels[17000:]

def patch_img(data, labels, nb_split):
	list_patch = []
	true_labs = []
	index_of_lab = 0

	for feature in data:
		patches_of_current_feature = np.split(feature, nb_split)

		for patch in patches_of_current_feature:
			list_patch.append(patch)
			true_labs.append(labels[index_of_lab])

		index_of_lab = index_of_lab + 1

	return np.asarray(list_patch), np.asarray(true_labs)

def make_new_features(data, centers, np_split):
	new_data = []

	for feature in data:
		patches_of_current_feature = np.split(feature, np_split)
		value_of_feature = []

		for patch in patches_of_current_feature:
			vector_of_patch = [0]*len(centers)
			distances = [np.linalg.norm(patch - centers[center]) for center in centers]
			cluster = distances.index(min(distances))
			vector_of_patch[cluster] = 1
			value_of_feature.extend(vector_of_patch)

		new_data.append(value_of_feature)

	return np.asarray(new_data, dtype="float64")


epoch_for_k_means = 20
epoch_for_perceptron = 300
nbr_cluster = 100


print("patching des images pour le train")
patches, true_labs = patch_img(data_train, labels_train, 4)

print("Création du classifier K_Means")
classifier_k_means = kMeans.K_Means(nbr_cluster, epoch_for_k_means)

print("Entrainement du classifier K_Means pour obtenir")
print("les centres des clusters entrainés")
classifier_k_means.fit(patches)
clusters, centers = classifier_k_means.classifications, classifier_k_means.centers

print("Patching des données de train et de test")
print("pour le passage au perceptron multi-classe")
new_data_train = make_new_features(data_train, centers, 4)
new_data_test = make_new_features(data_test, centers, 4)

print("Création du perceptron multi-classe")
classifier_perceptron = perceptronMC.perceptron_MC(
	new_data_train, new_data_test,
	labels_train, labels_test,
	epoch_for_perceptron)

print("Entrainement du perceptron multi-classe")
print("sur les nouveaux vecteurs de features + taux d'erreur :")
classifier_perceptron.fit()
