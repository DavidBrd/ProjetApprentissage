import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
			  [1.5, 1.8],
			  [5, 8],
			  [8, 8],
			  [3, 4],
			  [4, 6],
			  [1, 0.6],
			  [9, 11]])

plt.scatter(X[:,0], X[:,1], s=75)
plt.show()

colors = ["g", "r", "c", "b", "k"]

class K_Means:

	# Fonction d'initialisation : "k" : nombre de clusters, "tol" : seuil de tolérance permettant de savoir quand on arrête de mettre à jour les centres
	# "max_itération : nombre d'itération maximum pour la mise à jours des centres des clusters"
	def __init__(self, k, tol = 0.001, max_iteration = 300):
		self.k = k
		self.tol = tol
		self.max_iteration = max_iteration

	# Fonction d'optimisation prenant en entré le jeu de données
	def fit(self, data):

		# Initialisation du dictionnaire qui va contenir les centres, et les points associés à ces centres.
		self.centers = {}

		# Initialisation des centres par défaut avant optimisation (ceux-ci auraient pû être choisis au hasard cela n'a pas d'importance)
		for i in range(self.k):
			self.centers[i] = data[i]
		
		# Début de la boucle d'optimisation
		for i in range(self.max_iteration):

			# Déclaration d'un dictionnaire auxiliaire "classifications" qui va contenir les différents point en fontions des clusters.
			self.classifications = {}

			# Initialisation du dictionnaire, la valeur d'un clé est une liste qui va contenir les points.
			for j in range(self.k):
				self.classifications[j] = []

			# Ici, pour chaque feature du jeu de donnée, on calcul la distance de cette feature par rapport à chaque centre de cluster.
			for feature in data:
				distances = [np.linalg.norm(feature - self.centers[center]) for center in self.centers]
				classification = distances.index(min(distances))
				# On ajoute le point ayant la distance minimum au cluster correspondant.
				self.classifications[classification].append(feature)

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

# Création d'un classifier de la classe K_Means précédemment créée
classifier = K_Means(2)
# Entrainement du classifier
classifier.fit(X)

# Dessin des centres des clusters
for center in classifier.centers:
	plt.scatter(classifier.centers[center][0], classifier.centers[center][1], marker = "*", color = "k", s = 75)

# Affichage des coordonées des centres des clusters une fois le modèle entrainé.
for center in classifier.centers:
	print("coordonnées du centre du cluster numéro " + str(center) + " -> ", classifier.centers[center][0], classifier.centers[center][1])

# Dessin des points relatifs au clusters.
for classification in classifier.classifications:
	color = colors[classification]
	for feature in classifier.classifications[classification]:
		plt.scatter(feature[0], feature[1], color = color, s = 75)

plt.show()