
# Import
from sklearn.naive_bayes import GaussianNB
import csv

# To-do:
#   - Read file, input into two lists. Skip entries which contain missing values ("?"),  DONE
#   - How to attribute values in each instance to a number? What do we do for attributes with continuous values?
#   - How to choose split the data to train 5 separate classifiers? Consider class imbalance 
#   - What is the criteria to choose which class an instance belongs to? (i.e. confidence > 0.75,0.8,etc.)
#   - How do we average the result? 
#   - 

# Important Notes 
# - The string for each element in adult_data.txt begins with a space, i.e. "<=50K" is actually " <=50K" and "?" is actually " ?"
# - Make sure to exclude empty lists, as there exists some in the adult_data.txt file
# - Number of <=50K: 22654, Number of >50K: 7508 (Data without missing values, Approx. 3x difference) 

# Reading in data and separating it based on class. 
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

# --- Features and Feature Values ---
# age: continuous, mean = 38 
workFeatures = {' Private':1,' Self-emp-not-inc':2,' Self-emp-inc':3,' Federal-gov':4,' Local-gov':5,' State-gov':6,' Without-pay':7,' Never-worked':8}
# final-weight: continuous, mean = 189778
educationFeatures = {' Bachelors':1,' Some-college':2,' 11th':3,' HS-grad':4,' Prof-school':5,' Assoc-acdm':6,' Assoc-voc':7,' 9th':8,' 7th-8th':9,' 12th':10,' Masters':11,' 1st-4th':12,' 10th':13,' Doctorate':14,' 5th-6th':15,' Preschool':16}
maritalFeatures = {' Married-civ-spouse':1,' Divorced':2,' Never-married':3,' Separated':4,' Widowed':5,' Married-spouse-absent':6,' Married-AF-spouse':7}
# education-num: continuous, mean = 10
occupationFeatures = {' Tech-support': 0, ' Craft-repair': 1, ' Other-service': 2, ' Sales': 3, ' Exec-managerial': 4, ' Prof-specialty': 5, ' Handlers-cleaners': 6, ' Machine-op-inspct': 7, ' Adm-clerical': 8, ' Farming-fishing': 9, ' Transport-moving': 10, ' Priv-house-serv': 11, ' Protective-serv': 12, ' Armed-Forces': 13}
relationshipFeatures = {' Wife':1,' Own-child':2,' Husband':3,' Not-in-family':4,' Other-relative':5,' Unmarried':6}
raceFeatures = {' White':1,' Asian-Pac-Islander':2,' Amer-Indian-Eskimo':3,' Other':4,' Black':5}
sexFeatures = {' Female':1,' Male':2}
# capital-gain: continuous, mean = 1077
# capital-loss: continuous, mean = 87
# hours-per-week: continuous, mean = 40 
nativeFeatures = {' United-States': 0, ' Cambodia': 1, ' England': 2, ' Puerto-Rico': 3, ' Canada': 4, ' Germany': 5, ' Outlying-US(Guam-USVI-etc)': 6, ' India': 7, ' Japan': 8, ' Greece': 9, ' South': 10, ' China': 11, ' Cuba': 12, ' Iran': 13, ' Honduras': 14, ' Philippines': 15, ' Italy': 16, ' Poland': 17, ' Jamaica': 18, ' Vietnam': 19, ' Mexico': 20, ' Portugal': 21, ' Ireland': 22, ' France': 23, ' Dominican-Republic': 24, ' Laos': 25, ' Ecuador': 26, ' Taiwan': 27, ' Haiti': 28, ' Columbia': 29, ' Hungary': 30, ' Guatemala': 31, ' Nicaragua': 32, ' Scotland': 33, ' Thailand': 34, ' Yugoslavia': 35, ' El-Salvador': 36, ' Trinadad\\&Tobago': 37, ' Peru': 38, ' Hong': 39, ' Holand-Netherlands': 40}
classes = {' <=50K':1,' >50K':2}



combinedFeatures = {}
x_training = []
y_training = []    
print('finished')




# #transform the original training features to numbers and add them to the 4D array X.
# #For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# #transform the original training classes to numbers and add them to the vector Y.
# #For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# #--> add your Python code here
# for row in db: 
#   curRow = [] 
#   for i in range(0,len(row)):
#     if row[i] in features: 
#       curRow.append(features[row[i]])
#     elif row[i] in classes: 
#       Y.append(classes[row[i]])
#   X.append(curRow)

# #fitting the naive bayes to the data
# clf = GaussianNB()
# clf.fit(X, Y)

# #reading the test data in a csv file
# with open('weather_test.csv', 'r') as csvfile:
#   reader = csv.reader(csvfile)
#   for i, row in enumerate(reader):
#       if i > 0: #skipping the header
#          dbTest.append(row)

# #printing the header os the solution
# print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

# #use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
# testX = []
# for row in dbTest: 
#   curRow = [] 
#   for i in range(0,len(row)):
#     if row[i] in features: 
#       curRow.append(features[row[i]])
#   testX.append(curRow)

# for i in range(0,len(dbTest)):
#     confidence = list(clf.predict_proba([testX[i]])[0])
#     if confidence[0] > 0.75 or confidence[1] > 0.75:

#       if confidence[0] > 0.75: 
#         probability = confidence[0]
#         classInstance = 'Yes'
#       else: 
#         probability = confidence[1]
#         classInstance = 'No'
    

# ***For creating dictionaries***     
# with open('features.txt', 'r') as csvfile:
#   reader = csv.reader(csvfile)
#   for j,row in enumerate(reader):
#     if j == 1:
#       for i,element in enumerate(row):
#         nativeFeatures[element] = i
# print(nativeFeatures)













