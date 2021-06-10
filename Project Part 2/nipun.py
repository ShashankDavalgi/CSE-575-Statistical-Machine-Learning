import scipy.io, numpy as np,random, sys,math, matplotlib.pyplot as plt
from scipy.spatial import distance

inputFile=scipy.io.loadmat("AllSamples.mat")['AllSamples'] # this is for reading the data points which are 2 - dimensional
inputFileDimensions = inputFile.shape # this stores the dimension of the inputFile

centers = [] # this list stores the centroids which are randomly chosen from the data points
clustersCentroid={} # this is a dictionary which assigns a Cluster to every centroid
clustersPoints={} # this is a dictionary which stores all data points to the respective clusters
newcenters = [] # this is a dummy list which stores all centroids for successive iterations. It is used as a flag which is used to decide if the k means algorithm stops.
#newcenters = np.array(newcenters)

initialization = 0 # variable which stores the number of iterations.
x='c'
print("Please wait while K-means is being calculated for Strategy 1. It takes time to compute as there are many iterations until it finally converges")
while initialization <=1:
	J=[]
	for i in range(2,11):
		centers = [] # this list stores the centroids which are randomly chosen from the data points
		clustersCentroid={} # this is a dictionary which assigns a Cluster to every centroid
		clustersPoints={}
		newcenters = []
		randomGeneratorIndices = np.random.choice(inputFileDimensions[0],i,replace = False) # this is used to generate the indices randomly

		#this is used to assign random points as centers to centers list
		for j in randomGeneratorIndices:x
			centers.append(inputFile[j])
		#newcenters initially has None . It is used for updating centers on successive iterations
		for i in range(len(randomGeneratorIndices)):
			newcenters.append(None)
		newcenters = np.array(newcenters)
		count=1
		#this is used to store cluster centroid points to respective clusters
		for center in centers:
			clustersCentroid["Cluster"+str(count)] = center
			count=count+1
		centers = np.array(centers) # this is used for converting centers to numpy array for computation purposes
		intermedList=[]
		for key,val in clustersCentroid.items(): #this is used to assign centroid points to each and every cluster initially
			intermedList.append(list(val))
			clustersPoints[key] = intermedList
			intermedList=[]
		flag = False
		#the data points are assigned to clusters until the centroids don't change. The data points are assigned to the cluster with minimum euclidean distance from the respective point
		while not np.array_equal(centers,newcenters): # the stopping condition for the algorithm to stop when the centers remain same and not change.
			if flag: # this is used for re-initializing the clusterpoints and clustercentroids for successive iterations until the algorithm converges
				clustersCentroid={}
				centers = newcenters
				count=1
				for center in centers:
					clustersCentroid["Cluster"+str(count)] = center
					count=count+1
				clustersPoints={}
				intermedList=[]
				for key,val in clustersCentroid.items():
					intermedList.append(list(val))
					clustersPoints[key] = intermedList
					intermedList=[]
			#lines 59 to 77 are used for assigning data points to respective clusters
			for data in inputFile:
				if data not in centers:
					minimum = sys.maxsize
					minCenter = sys.maxsize
					for center in centers:
						dist=distance.euclidean(data,center) #the data points are assigned to the cluster with the smallest euclidean distance.
						if dist < minimum:
							minimum = dist
							minCenter = center
					for key, val in clustersCentroid.items():
						if str(val) == str(minCenter):
							if key not in clustersPoints.keys():
								dataPts = []
								dataPts.append(list(data))
								clustersPoints[key] = dataPts
							else:
								dataPts = clustersPoints[key]
								dataPts.append(list(data))
								clustersPoints[key]=dataPts
			newcenters=[]
			for key,val in clustersPoints.items(): # this is the code for updating the centers after assigning all the data points to the respective clusters
				res=np.mean(val,axis=0)
				newcenters.append(res)
			newcenters = np.array(newcenters) # this is used for converting newcenters to numpy array for computation purposes
			flag = True
		index = 0
		for key,val in clustersCentroid.items(): # this is for updating the clusterscentroid with new centers for successive iterations
			clustersCentroid[key] = centers[index]
			index+=1

		#code for calculating objective function
		sse = 0
		for key,val in clustersPoints.items():
			centerKeyVal = list(clustersCentroid.get(key))
			for value in val:
				dist = distance.euclidean(value,centerKeyVal)
				distSquared = math.pow(dist,2)
				sse+=distSquared
		J.append(sse)

	print("Objective function is %s" % J)

	#code for plotting graphs
	K=[k for k in range(2,11)]
	plt.title("K-Means Strategy-1")
	l='Initialization-'+str(initialization+1)
	plt.ylabel('Objective Function J(k)')
	plt.xlabel('number of clusters k')
	plt.plot(K,J,x,marker='o',label=l)
	plt.legend()
	print("Objective function values Initialization:"+str(initialization+1))
	print("K:"+str(K))
	print("J(K):"+str(J))
	print("\n")
	plt.show()
	initialization+=1
	x='black'
	#z=5
plt.show()
##################################################   End of strategy - 1  ###############################################################################################################

##################################################### Strategy - 2 beginning ##################################################################



import scipy.io, numpy as np,random, sys,math, matplotlib.pyplot as plt
from scipy.spatial import distance

inputFile=scipy.io.loadmat("AllSamples.mat")['AllSamples']

inputFileDimensions = inputFile.shape

initialization = 0
x = 'c'
print("Please wait while K-means is being calculated for Strategy 2. It takes time to compute as there are many iterations until it finally converges")
while initialization <= 1:
	J=[]
	for k in range(2,11):
		centers = [] # this list stores the centroids which are randomly chosen from the data points
		clustersCentroid={} # this is a dictionary which assigns a Cluster to every centroid
		clustersPoints={} # this is a dictionary which stores all data points to the respective clusters
		newcenters = [] # this is a dummy list which stores all centroids for successive iterations. It is used as a flag which is used to decide if the k means algorithm stops.
		randomGeneratorIndices = np.random.choice(inputFileDimensions[0],1,replace = False) # this is used to generate just 1 index randomly
		centers.append(list(inputFile[randomGeneratorIndices][0]))
		#this is the differentiating part of the code where the centers are assigned based on strategy 2 (lines 145 to 159)
		while len(centers)<k:

			maximum = -sys.maxsize - 1
			maxCenter = -sys.maxsize - 1
			for i in inputFile:
				dist=0
				for j in centers:
					if list(i) not in centers:

						dist += distance.euclidean(i,j) # the distances to the previous i-1 centers is summed
				dist=dist/len(centers) # average value of sum of distances to previous i-1 centers is taken
				if dist>maximum:
					maximum=dist
					maxCenter = i
			centers.append(list(maxCenter)) # the point with maximum average distance to previous i-1 centers is chosen as the next center

		for i in range(len(randomGeneratorIndices)): #newcenters is initialized to None
			newcenters.append(None)
		newcenters = np.array(newcenters) # this is used for converting newcenters to numpy array for computation purposes
		count=1
		for center in centers:
			clustersCentroid["Cluster"+str(count)] = center
			count=count+1
		centers = np.array(centers) # this is used for converting centers to numpy array for computation purposes

		intermedList=[]
		for key,val in clustersCentroid.items():
			intermedList.append(list(val))
			clustersPoints[key] = intermedList
			intermedList=[]
		flag = False
		#the data points are assigned to clusters until the centroids don't change. The data points are assigned to the cluster with minimum euclidean distance from the respective point
		while not np.array_equal(centers,newcenters):  # the stopping condition for the algorithm to stop when the centers remain same and not change.
			if flag: # this is used for re-initializing the clusterpoints and clustercentroids for successive iterations until the algorithm converges
				clustersCentroid={}
				centers = newcenters
				count=1
				#this is used to store cluster centroid points to respective clusters
				for center in centers:
					clustersCentroid["Cluster"+str(count)] = center
					count=count+1
				clustersPoints={}
				intermedList=[]
				for key,val in clustersCentroid.items(): #this is used to assign centroid points to each and every cluster initially
					intermedList.append(list(val))
					clustersPoints[key] = intermedList
					intermedList=[]
			#lines 191 to 210 are used for assigning data points to respective clusters
			for data in inputFile:
				if data not in centers:
					minimum = sys.maxsize
					minCenter = sys.maxsize
					for center in centers:
						dist=distance.euclidean(data,center) #the data points are assigned to the cluster with the smallest euclidean distance.
						if dist < minimum:
							minimum = dist
							minCenter = center
					for key, val in clustersCentroid.items():
						if list(val) == list(minCenter):
							if key not in clustersPoints.keys():
								dataPts = []
								dataPts.append(list(data))
								clustersPoints[key] = dataPts
							else:
								dataPts = clustersPoints[key]
								dataPts.append(list(data))
								clustersPoints[key]=dataPts
			newcenters=[]
			for key,val in clustersPoints.items(): # this is the code for updating the centers after assigning all the data points to the respective clusters
				res=np.mean(val,axis=0)
				newcenters.append(res)
			newcenters = np.array(newcenters) # this is used for converting newcenters to numpy array for computation purposes
			flag = True
		index = 0
		for key,val in clustersCentroid.items(): # this is for updating the clusterscentroid with new centers for successive iterations
			clustersCentroid[key] = centers[index]
			index+=1
		#code for calculating objective function
		sse = 0
		for key,val in clustersPoints.items():
			centerKeyVal = list(clustersCentroid.get(key))
			for value in val:
				sse+=math.pow(distance.euclidean(value,centerKeyVal),2)
		J.append(sse)
	#code for plotting graphs
	print("Objective function is %s" % J)
	K=[k for k in range(2,11)]
	plt.title("K-Means Strategy-2")
	l='Initialization-'+str(initialization+1)
	plt.ylabel('Objective Function J(k)')
	plt.xlabel('number of clusters k')
	plt.plot(K,J,x,marker='o',label=l)
	plt.legend()
	print("Objective function values Initialization:"+str(initialization+1))
	print("K:"+str(K))
	print("J(K):"+str(J))
	print("\n")
	plt.show()
	initialization+=1
	x='green'
plt.show()
