import numpy as np

class K_Means:

	# Fonction d'initialisation : "k" : nombre de clusters, "tol" : seuil de tolérance permettant de savoir quand on arrête de mettre à jour les centres
	# "max_itération : nombre d'itération maximum pour la mise à jours des centres des clusters"
	def __init__(self, k, epoch):
		self.k = k
		self.epoch = epoch

	# Fonction d'optimisation prenant en entré le jeu de données
	def fit(self, data):

		# Initialisation du dictionnaire qui va contenir les centres, et les points associés à ces centres.
		self.centers = {}
		self.idx_centers = [[] for i in range(self.k)]

		# Initialisation des centres par défaut avant optimisation (ceux-ci auraient pû être choisis au hasard cela n'a pas d'importance)
		for i in range(self.k):
			self.centers[i] = data[i]
		
		# Début de la boucle d'optimisation
		for i in range(self.epoch):

			# Déclaration d'un dictionnaire auxiliaire "classifications" qui va contenir les différents point en fontions des clusters.
			self.classifications = {}

			# Initialisation du dictionnaire, la valeur d'un clé est une liste qui va contenir les points.
			for j in range(self.k):
				self.classifications[j] = []

			# Ici, pour chaque feature du jeu de donnée, on calcul la distance de cette feature par rapport à chaque centre de cluster.
			for i, feature in enumerate(data):
				distances = [np.linalg.norm(feature - self.centers[center]) for center in self.centers]
				classification = distances.index(min(distances))
				# On ajoute le point ayant la distance minimum au cluster correspondant.
				self.classifications[classification].append(feature)
				self.idx_centers[classification].append(i)

			# On garde un trace des centres des clusters avant de faire la mise à jour.
			prev_centers = dict(self.centers)

			# C'est ici que l'on met à jour la position des centres des clusters en faisant la moyen avec les coordonnées des points pour chacun des clusters.
			for classification in self.classifications:
				self.centers[classification] = np.average(self.classifications[classification], axis = 0)

			# Petit boucle permettant de décider de l'arrêt de la mise à jour à partir d'un certain seuil.
			# Ce code n'est pas indispensable au sens où l'on à déjà un nombre d'itération maximum.
			# optimized = True
			# for c in self.centers:
			# 	original_center = prev_centers[c]
			# 	new_center = self.centers[c]
			# 	if np.sum((new_center - original_center) / new_center*100.0) > self.tol:
			# 		optimized = False

			# if optimized:
			# 	break

	# Fonction de prédiction permettant de demandé au modèle une prédiction sur une donnée en particulier
	def predict(self, data):
		distances = [np.linalg.norm(data - self.centers[center]) for center in self.centers]
		classification = distances.index(min(distances))
		return classification

	def eval_k_means(self, labels):
		for cluster in self.classifications:

			classes = [0 for x in range(10)]
			# Calcul de la classe dominante
			for j in range(len(self.classifications[cluster])):
				classes[ labels[self.idx_centers[cluster][j]] ] += 1

			# On détermine quelle classe est la plus représentée
			index_of_bigger_class = 0
			max_value_of_class = classes[0]

			for j in range (1, len(classes)):
				if (classes[j] > max_value_of_class):
					index_of_bigger_class = j
					max_value_of_class = classes[j]

			error_rate = 0

			for j in range(len(self.classifications[cluster])):
				if (index_of_bigger_class != labels[self.idx_centers[cluster][j]]):
					error_rate += 1
			error_rate = error_rate / len(self.classifications[cluster])
			print("cluster [", cluster, "] (", len(self.classifications[cluster]), ") : taux erreurs = ", error_rate)


# classifier = K_Means(10)
# classifier.fit(data)
# classifier.eval_k_means(labels)