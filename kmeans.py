import random
import math
import numpy as np

# Computes euclidian distance
def dist(A, B):
  s = 0.
  for a, b in zip(A, B): s += (a - b) ** 2
  return math.sqrt(s)

def eq(A, B):
  for a, b in zip(A, B):
    if a != b:
      return False
  return True

# Chooses initial barycenters randomly
def choose_initial(data, k):
  return random.sample(data, k)

# Returns the index of the closest neighbor of a in l
def closest_neighbor(a, l):
  # Init
  current_i = 0
  min_dist = dist(a, l[current_i])

  # Iterating by index (we'll need it)
  for i in range(0, len(l)):
    e = l[i]
    d = dist(a, e)
    if(d < min_dist):
      current_i = i
      min_dist = d
  return current_i

# Extracts all the points from a given cluster
def extract_cluster(k, data, cluster_ids):
  res = []
  for i in range(0, len(data)):
    if cluster_ids[i] == k: res.append(data[k])
  return res

# Computes a barycenter
def barycenter(cluster):
  bary = list(cluster[0])
  for i in range(1, len(cluster)):
    for j in range(0, len(bary)):
      bary[j] += cluster[i][j]

  for i in range(0, len(bary)):
    bary[i] /= len(cluster)
  return bary

# Computes an error rate for a center and a given cluster
def cluster_error(center, cluster):
  err = 0.
  for e in cluster:
    err += dist(center, e)
  return err

def kmeans(data, k, t, maxiter):
  # Choosing the first clusters
  cluster_centers = choose_initial(data, k)

  # Assigning a cluster to each point of data
  cluster_ids = [closest_neighbor(d, cluster_centers) for d in data]

  for it in range(0, maxiter):
    print("Iteration #", it, "...")

    update  = False # Update bit
    error   = 0.    # Error rate

    for i in range(0, len(cluster_centers)):
      print("\tCluster #", i, "...")

      # Extract the damn cluster !
      cluster = extract_cluster(i, data, cluster_ids)
      if(len(cluster) == 0): continue # lol if empty

      c     = cluster_centers[i]  # Old center
      new_c = barycenter(cluster) # Updated center
      
      error  += cluster_error(new_c, cluster)

      # Flips update if a barycenter was updated
      if not eq(c, new_c):
        cluster_centers[i] = new_c
        update = True

    # Break loop if the error is low enough or there is no update
    if error < t or update == False: break

    # Compute new cluster ids
    cluster_ids = [closest_neighbor(d, cluster_centers) for d in data]

  return cluster_ids, cluster_centers
