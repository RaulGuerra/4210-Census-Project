#-------------------------------------------------------------------------
# AUTHOR: Raul Guerra
# FILENAME: bagging_random_forest.py
# SPECIFICATION: builds a base classifier by using a single decision tree, an ensemble classifier that combines multiple decision trees, and a Random Forest
# classifier to recognize handwritten digits
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dataTraining = 'optdigits.tra'
dataTest = 'optdigits.tes'
dbTraining = []
dbTest = []
X_test = []
y_test = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
with open(dataTraining, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        dbTraining.append (row)


#reading the test data from a csv file and populate dbTest
#--> add your Python code here
with open(dataTest, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        dbTest.append (row)


#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
for i in range(1797):
    classVotes.append([0,0,0,0,0,0,0,0,0,0])
#classVotes.append([0,0,0,0,0,0,0,0,0,0])

print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  #--> add your Python code here
  X_training = [list(map(int, i)) for i in bootstrapSample]
  y_training.clear()
  for sub_list in X_training:
      y_training.append(sub_list.pop(-1))

  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)
  accuratePredictions = 0
  totalPredictions = 0
  iterator = 0

  #for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here

  X_test = [list(map(int, i)) for i in dbTest]
  for sub_list in X_test:
      y_test.append(sub_list.pop(-1))

  n = 0
  for sub_list in X_test:
      class_predicted = clf.predict([sub_list])[0]
      totalPredictions += 1
      if class_predicted == y_test[iterator]:
          j = class_predicted
          classVotes[n][j] += 1
          accuratePredictions += 1
      iterator += 1
      n += 1




  if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
      #--> add your Python code here
      accuracy = accuratePredictions/totalPredictions

  if k == 0: #for only the first base classifier, print its accuracy here
      #--> add your Python code here
      print("Finished my base classifier (fast but relatively low accuracy) ...")
      print("My base classifier accuracy: " + str(accuracy))
      print("")

#now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with
#the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
#--> add your Python code here
accuratePredictions = 0
totalPredictions = 0
for i in range(1797):
    if y_test[i] == classVotes[i].index(max(classVotes[i])):
        accuratePredictions += 1
    totalPredictions += 1

accuracy = accuratePredictions/totalPredictions


#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before




#Fit Random Forest to the training data

X_training = [list(map(int, i)) for i in dbTraining]
y_training.clear()
for sub_list in X_training:
    y_training.append(sub_list.pop(-1))

clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here

iterator = 0
accuratePredictions = 0
totalPredictions = 0
for sub_list in X_test:
    class_predicted = clf.predict([sub_list])[0]
    totalPredictions += 1
    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here
    if class_predicted == y_test[iterator]:
        accuratePredictions += 1
    iterator += 1


    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here

accuracy = accuratePredictions/totalPredictions


#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
