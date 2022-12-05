
# Import
from sklearn.naive_bayes import GaussianNB
import csv
import random

# To-do:
#   - Read file, input into two lists. Skip entries which contain missing values ("?"),  DONE
#   - How to attribute values in each instance to a number? What do we do for attributes with continuous values? DONE
#   - How to choose split the data to train 5 separate classifiers? Consider class imbalance IN PROGRESS
#         - Implement Random Sampling (No Duplicates) From Both Classes for the data used in each classifier (4k each)
#   - What is the criteria to choose which class an instance belongs to? (i.e. confidence > 0.75) IN PROGRESS
#         - Current criteria: 0.75 
#   - How do we average the result? IN PROGRESS

# Important Notes 
# - The string for each element in adult_data.txt begins with a space, i.e. "<=50K" is actually " <=50K" and "?" is actually " ?"
# - Make sure to exclude empty lists, as there exists some in the adult_data.txt file
# - Number of <=50K: 22654, Number of >50K: 7508 (Data without missing values, Approx. 3x difference) 
# - Result for a single naive-bayes classifier had an accuracy of 74.44% (7509 Sample/Class), using ensemble we should get a greater value 
# - Compared bootstrap sampling (replacement) to just randomly selecting it from the pool of samples 
  # - Lower confidence doesn't really affect (very slightly) the average accuracy of the model  
  # - Bootstrap sampling actually performs slightly worse than if it is randomly sampled without replacement 
# - Compared continuous attributes to having discrete attriubtes for naive bayes 
# - Ensemble for naive-bayes generally remains consistent with a single naive-bayes model

# Reading in training data and separating it based on class. 
incomeDataGreat = []
incomeDataLess = []
with open('adult_data.txt', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
    if row and not ' ?' in row:
      if row[14] == " <=50K":
        incomeDataLess.append(row)
      else:
        incomeDataGreat.append(row)

# Reading testing data 
incomeTestData = []
incomeTestResult = []
with open('adult_test.txt', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
    if row and not ' ?' in row:
      incomeTestData.append(row[0:14])
      incomeTestResult.append(row[14])

# --- Features and Feature Values ---
# age: continuous, mean = 38 
work = {' Private':1,' Self-emp-not-inc':2,' Self-emp-inc':3,' Federal-gov':4,' Local-gov':5,' State-gov':6,' Without-pay':7,' Never-worked':8}
# final-weight: continuous, mean = 189778
education = {' Bachelors':1,' Some-college':2,' 11th':3,' HS-grad':4,' Prof-school':5,' Assoc-acdm':6,' Assoc-voc':7,' 9th':8,' 7th-8th':9,' 12th':10,' Masters':11,' 1st-4th':12,' 10th':13,' Doctorate':14,' 5th-6th':15,' Preschool':16}
# education-num: continuous, mean = 10
marital= {' Married-civ-spouse':1,' Divorced':2,' Never-married':3,' Separated':4,' Widowed':5,' Married-spouse-absent':6,' Married-AF-spouse':7}
occupation = {' Tech-support': 0, ' Craft-repair': 1, ' Other-service': 2, ' Sales': 3, ' Exec-managerial': 4, ' Prof-specialty': 5, ' Handlers-cleaners': 6, ' Machine-op-inspct': 7, ' Adm-clerical': 8, ' Farming-fishing': 9, ' Transport-moving': 10, ' Priv-house-serv': 11, ' Protective-serv': 12, ' Armed-Forces': 13}
relationship = {' Wife':1,' Own-child':2,' Husband':3,' Not-in-family':4,' Other-relative':5,' Unmarried':6}
race= {' White':1,' Asian-Pac-Islander':2,' Amer-Indian-Eskimo':3,' Other':4,' Black':5}
sex = {' Female':1,' Male':2}
# capital-gain: continuous, mean = 1077
# capital-loss: continuous, mean = 87
# hours-per-week: continuous, mean = 40 
native = {' United-States': 0, ' Cambodia': 1, ' England': 2, ' Puerto-Rico': 3, ' Canada': 4, ' Germany': 5, ' Outlying-US(Guam-USVI-etc)': 6, ' India': 7, ' Japan': 8, ' Greece': 9, ' South': 10, ' China': 11, ' Cuba': 12, ' Iran': 13, ' Honduras': 14, ' Philippines': 15, ' Italy': 16, ' Poland': 17, ' Jamaica': 18, ' Vietnam': 19, ' Mexico': 20, ' Portugal': 21, ' Ireland': 22, ' France': 23, ' Dominican-Republic': 24, ' Laos': 25, ' Ecuador': 26, ' Taiwan': 27, ' Haiti': 28, ' Columbia': 29, ' Hungary': 30, ' Guatemala': 31, ' Nicaragua': 32, ' Scotland': 33, ' Thailand': 34, ' Yugoslavia': 35, ' El-Salvador': 36, ' Trinadad&Tobago': 37, ' Peru': 38, ' Hong': 39, ' Holand-Netherlands': 40}
classes = {' <=50K':1,' >50K':2}

# Combine all the discrete features into a single dictionary
combinedFeatures = work | education | marital | occupation | relationship | race | sex | native

# Mapping values of the testing data
testX = []
continousAttributeIndexes = [0,2,4,10,11,12]
for j,row in enumerate(incomeTestData):
  curRow = [] 
  for i,element in enumerate(row):
    if i in continousAttributeIndexes:
        curRow.append(int(element))
    elif element in combinedFeatures:
        curRow.append(combinedFeatures[element])
  testX.append(curRow)

# Var
totalAcc = 0 
X = []
Y = []
x_training = []

# Obtain accuracy of 5 seperate classifiers built using randomized data, averages accuracy from all 5 models and returns it  
for k in range(0,5):

  # Clear from previous iterations
  X.clear
  Y.clear
  x_training.clear()

  # Random Sampling (4k points from each class)
  x_training.extend(random.sample(incomeDataLess,k=4000))
  x_training.extend(random.sample(incomeDataGreat,k=4000))
  # - Other Ways of Sampling Data -
  # x_training.extend(resample(incomeDataGreat, n_samples=7508, replace=True))
  # x_training.extend(resample(incomeDataLess, n_samples=7508, replace=True))
  # x_training = incomeDataGreat
  # x_training.extend(incomeDataLess)
  
  # Mapping values to each attribute in each row of the training data, transforming the data into numbers 
  for row in x_training:
    curRow = [] 
    for i,element in enumerate(row):
        if element in combinedFeatures:
          curRow.append(combinedFeatures[element])
        elif element in classes:
          Y.append(classes[element])
        else:
          curRow.append(int(element))
    X.append(curRow)
    
  # Fitting naive bayes to the data
  clf = GaussianNB()
  clf.fit(X, Y)

  # For each test sample, run it through the naive_bayes model to get the confidence (form of [prob1,prob2]) 
  # Then compare against decision criteria (> 0.75) to predict the class (confidence[0] = "<=50K", confidence[1] = ">50K")
  # Compare the prediction to the actual test class, calculate accuracy. 
  predictionsCorrect = 0
  for i, test in enumerate(testX):
    confidence = list(clf.predict_proba([test])[0])

    if confidence[0] > 0.75: 
      instanceClass = ' <=50K'
    else: 
      instanceClass = ' >50K'
    
    actualClass = incomeTestResult[i][0:len(incomeTestResult[i])-1]
    if instanceClass == actualClass:
      predictionsCorrect += 1
      
  accuracy = predictionsCorrect / len(testX)
  totalAcc += accuracy
  print("Accuracy of classifier #" + str(k+1) + ": " + str(accuracy))

print("\nAverage Accuracy of All Classifiers:",(totalAcc/5))

print('finished')

# # --- FOR TESTING PURPOSES ---
# # Using 15018 Testing Samples (7509 ">50K" / 7509 "<=50K")
# x_training = incomeDataGreat[0:7509]
# x_training.extend(incomeDataLess)
# X = []
# Y = [] 
# # Mapping values to each attribute in each row of the training data, transforming the data into numbers 
# for row in x_training:
#   curRow = [] 
#   for i,element in enumerate(row):
#     if i == 0:
#       if int(element) > 38:
#         curRow.append(1)
#       else:
#         curRow.append(2)
#     elif i == 2:
#       if int(element) > 189778:
#         curRow.append(1)
#       else:
#         curRow.append(2)
#     elif i == 4:
#       if int(element) > 10:
#         curRow.append(1)
#       else:
#         curRow.append(2) 
#     elif i == 10:
#       if int(element) > 1077:
#         curRow.append(1)
#       else:
#         curRow.append(2)
#     elif i == 11:
#       if int(element) > 87:
#         curRow.append(1)
#       else:
#         curRow.append(2) 
#     elif i == 12:
#       if int(element) > 40:
#         curRow.append(1)
#       else:
#         curRow.append(2) 
#     elif element in combinedFeatures:
#       curRow.append(combinedFeatures[element])
#     else:
#       Y.append(classes[element])
#   X.append(curRow)

# --- FOR CREATING DICTIONARIES ---
# with open('features.txt', 'r') as csvfile:
#   reader = csv.reader(csvfile)
#   for j,row in enumerate(reader):
#     if j == 1:
#       for i,element in enumerate(row):
#         nativeFeatures[element] = i
# print(nativeFeatures)














