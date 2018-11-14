# 1005031
# Arif Uz Zaman
# CSE,BUET

import math
import random
import operator
from timeit import default_timer as timer


trainData = None
testData = None

Words = set([])
ignore = ['','a','an','the','A','An','The']

# File Input
def fileInput():
	global trainData,testData

	with open("training.data","r+") as fhandle:
		Train = fhandle.read().replace("\n\n\n\n\n","\n\n\n").replace("\n\n\n\n","\n\n\n")
		trainData = [[ topic for topic in article.split("\n\n")] for article in Train.split("\n\n\n")]

	with open("test.data","r+") as fhandle:
		Test = fhandle.read().replace("\n\n\n\n\n","\n\n\n").replace("\n\n\n\n","\n\n\n")
		testData = [[ topic for topic in article.split("\n\n")] for article in Test.split("\n\n\n")]


def inputStats():
	print ("Number of Article - Train,Test : %d %d") % (len(trainData),len(testData))
	print

	temp1 = []
	temp1 = set(temp1)
	for x in xrange(len(trainData)):
		temp1.add(trainData[x][0])

	temp1 = set(sorted(temp1))

	temp2 = []
	temp2 = set(temp2)
	for x in xrange(len(testData)):
		temp2.add(testData[x][0])

	temp2 = set(sorted(temp2))

	print "Total Tag in Train : ",len(temp1)
	print temp1
	print

	print "Total Tag in Test : ",len(temp2)
	print temp2
	print

	print "Common Tag : ",len( set(temp1).intersection(temp2))
	print set(temp1).intersection(temp2)
	print

	print "Only in Train : ",len(temp1-temp2)
	print temp1-temp2
	print

	print "Only in Test",len(temp2-temp1)
	print temp2-temp1


	count = 0
	common = set(temp1).intersection(temp2)
	for x in xrange(len(testData)):
		if testData[x][0] in common:
			count += 1

	print
	print ("Target : %3.1f") % (count/float(len(testData))*100)


# Vocabulary
def uniqueWord():
	global Words

	count = 0
	for x in xrange(len(trainData)):
		for y in xrange(1,len(trainData[x])):
			str1 = trainData[x][y].replace("\n"," ")
			for word in str1.split(" "):
				count += 1
				if word in ignore:
					continue

				Words.add(word)

	#'''
	for x in xrange(len(testData)):
		for y in xrange(1,len(testData[x])):
			str1 = testData[x][y].replace("\n"," ")
			for word in str1.split(" "):
				count += 1
				if word in ignore:
					continue

				Words.add(word)

	#print len(Words),count
	#'''


def featureVector(matrix):
	global Words
	vector = [ 0 for x in xrange(len(Words)) ]

	temp = {}
	for x in xrange(1,len(matrix)):
		str1 = matrix[x].replace("\n"," ")

		for word in str1.split(" "):
			if word in ignore:
				continue

			if word in temp:
				temp[word] += 1
			else:
				temp[word] = 1

	Word = list(Words)
	for key in temp:
		for x in xrange(len(Word)):
			if key == Word[x]:
				vector[x] = temp[key]
				break

	return vector


def hDistance(v1,v2):
	hd = 0
	for x in xrange(len(v1)):
		if (v1[x] == 0 and v2[x] !=0) or (v1[x] != 0 and v2[x] ==0):
			hd += 1

	return hd


def hammingDistance(k):
	nn = [[ 0 for x in xrange(k) ] for y in xrange(len(testData)) ]

	for x in xrange(len(testData)):
		distance = []
		for y in xrange(len(trainData)):
			distance.append(hDistance(testVectors[x],trainVectors[y]))

		for y in xrange(k):
			indx = distance.index(min(distance))
			nn[x][y] = indx
			distance[indx] = max(distance)

	return nn


def eDistance(v1,v2):
	ed = 0
	for x in xrange(len(v1)):
		ed += (v1[x]-v2[x]) ** 2

	return math.sqrt(ed)


def euclideanDistance(k):
	nn = [[ 0 for x in xrange(k) ] for y in xrange(len(testData)) ]

	for x in xrange(len(testData)):
		distance = []
		for y in xrange(len(trainData)):
			distance.append(eDistance(testVectors[x],trainVectors[y]))

		for y in xrange(k):
			indx = distance.index(min(distance))
			nn[x][y] = indx
			distance[indx] = max(distance)

	return nn


def weightedVector():
	global trainVectors,testVectors

	D1,D2, = len(trainVectors), len(testVectors)
	D = max(D1,D2)

	for x in xrange(D):
		for y in xrange(len(Words)):
			if x<D1:
				if trainVectors[x][y] == 0:
					trainVectors[x][y] = 0.001
				else:
					Cw,Nw,wd = 0,trainVectors[x][y],sum(trainVectors[x])
					tf = Nw/float(wd)
					for z in xrange(D1):
						if trainVectors[z][y] != 0:
							Cw += 1

					idf = math.log10(D1/float(Cw))
					trainVectors[x][y] = tf*idf

			if x<D2:
				if testVectors[x][y] != 0:
					Cw,Nw,wd = 0,testVectors[x][y],sum(testVectors[x])
					tf = Nw/float(wd)
					for z in xrange(D2):
						if testVectors[z][y] != 0:
							Cw += 1

					idf = math.log10(D2/float(Cw))
					testVectors[x][y] = tf*idf


def cSimilarity(v1,v2):
	XX,YY,XY = 0,0,0

	for i in xrange(len(v1)):
		x,y = v1[i],v2[i]
		XX += x*x
		YY += y*y
		XY += x*y

	return XY/math.sqrt(XX*YY)


def cosineSimilarity(k):
	nn = [[ 0 for x in xrange(k) ] for y in xrange(len(testData)) ]

	for x in xrange(len(testData)):
		similarity = []
		for y in xrange(len(trainData)):
			similarity.append(cSimilarity(testVectors[x],trainVectors[y]))

		for y in xrange(k):
			indx = similarity.index(max(similarity))
			nn[x][y] = indx
			similarity[indx] = min(similarity)

	return nn


def nearestNeighbours(k,n):
	if n==0:
		start = timer()
		neighbours = hammingDistance(k)
		end = timer()
		print ("\tHamming Distance : %3.3f sec") % (end - start)

	elif n==1:
		start = timer()
		neighbours = euclideanDistance(k)
		end = timer()
		print ("\tEuclidean Distance : %3.3f sec") % (end - start)

	else:
		start = timer()
		weightedVector()
		end = timer()
		print ("\tTF-IDF : %3.3f sec") % (end - start)

		start = timer()
		neighbours = cosineSimilarity(k)
		end = timer()
		print ("\tCosine Similarity : %3.3f sec") % (end - start)

	return neighbours


def findClass(nList,k):
	classList = []

	for x in xrange(len(nList)):
		classVotes = {}

		for y in xrange(k):
			response = nList[x][y]

			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1

		indx = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
		classList.append(trainData[indx][0])

	return classList


def testFindings(nList,k):
	classList,count = findClass(nList,k),0

	for x in xrange(len(classList)):
		if testData[x][0] == classList[x]:
			count += 1

	return count/float(len(testData)) * 100


def offline(k):
	global trainVectors,testVectors

	accuracy = [[ 1 for j in xrange(3) ] for i in xrange(k/2 + 1)]
	trainVectors = [ featureVector(trainData[x]) for x in xrange(len(trainData)) ]
	testVectors = [ featureVector(testData[x]) for x in xrange(len(testData)) ]

	for x in xrange(3):
		neighbours = nearestNeighbours(k,x)
		for y in xrange(k/2 + 1):
			succ = testFindings(neighbours,2*y+1)
			accuracy[y][x] = succ

	print
	print "Output Table :"
	print "\t K \t Hamming \t Euclidean \t Cosine"

	for x in xrange(1,k+1,2):
		print ("\t %d \t %4.2f \t\t %4.2f \t\t %4.2f") % (x,accuracy[x/2][0],accuracy[x/2][1],accuracy[x/2][2])

	with open("Output.txt","w+") as fhandle:
		for line in accuracy:
			for word in line:
				word = str(word)[:5]
				fhandle.write(word)
				fhandle.write("\t")

			fhandle.write("\n")


def main(num):
	random.seed(num)
	global trainData,testData

	print "Offline : 02 - K Nearest Neighbours Algorithm..."
	fileInput()
	random.shuffle(trainData)
	random.shuffle(testData)
	trainData,testData = trainData[:100],testData[:100]

	inputStats()
	print
	uniqueWord()
	print "Run Time :"
	#offline(5)


if __name__ == '__main__':
	start = timer()
	main(136)
	end = timer()
	print
	print ("Program Run Time = %3.2f min") % ((end - start)/60)