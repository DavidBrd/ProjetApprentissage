import pickle

# Chargement du fichier de données à l'aide de pickle.
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

# "full_dataset" contient le fichier brut, il est traité par la suite.
full_dataset = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')

# Split des données en données de training, et données de validation.
# De même pour les labels
data = full_dataset[b'data']
labels = full_dataset[b'labels']
data_train = data[:9000]
data_valid = data[9000:]
labels_train = labels[:9000]
labels_valid = labels[9000:]

label_names = unpickle('./cifar-10-python/cifar-10-batches-py/batches.meta')[b'label_names']
name_img = full_dataset[b'filenames']

# Dictionnaire permettant de voir quel numéro de label est associé à quelle catégorie d'image.
label_correspondance = {}
for l in labels:
	label_correspondance[l] = label_names[l]

# Initialisaiton d'un tableau "repartition" permettant de voir le nombre d'exemple par classe ici dans le batch 1 du jeu de données.
repartition = [0,0,0,0,0,0,0,0,0,0]
for i in labels:
 	repartition[i] += 1
for i in range(10):
	repartition[i] = repartition[i] / len(labels)

# print(data)
# print(labels)
# print(label_names)
# print(name_img)
# print(label_correspondance)
# print(repartition)

# --------------------------------------------------------------------------------------------------------------

# Il y a 10000 images représentées dans les différents batch (ici le batch 1)
# Pour ces 10000 images il y a 10 classes différentes pour la classification
# On peut facilement regarder la répartition des exemples par classe, il y a environ 10% pour chacune des classes, la distribution est relativement homogène.
# Voir "repartition" contenant précisémment la repartition des images par classes.
# Dans Cifar-10 les classes sont assimilées au catégories suivantes : automobile, bird, cat, deer, dog, frog, horse, ship, truck.
# On peut noter que la tache de classification pour un humain est relativement facile. Cependant pour une machine cela reste beaucoup plus complexe.
# Pourquoi pas réaliser un approche naïve avec un algorithm tel que K-Means, et ainsi analyser les résultats obtenus.