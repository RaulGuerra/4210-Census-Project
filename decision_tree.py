#-------------------------------------------------------------------------
# AUTHOR: Raul Guerra
# FILENAME: decision_tree.py
# SPECIFICATION: Creates a decision tree based on a given CSV file.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
dbTest = []
X = []
Xtest =[]
Y = []
Ytest = []
columns = 15 #might give error. May need to be 14. Since 0-14
rows = 32561 #same as above
rowsTest = 16281

#attributes
workingClass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']
education = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?']
relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex = ['Male', 'Female']
nativeCountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
income = ['<=50K', '>50K']


#reading the data in a csv file
with open('adult.DATA', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         #print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

#--> add your Python code here

for i in range(rows):
    sub = []
    for j in range(columns):

        if j == 0: #checking age
            if int(db[i][j]) < 38:
                sub.append(1)
                #X[i][j] = 1 #if age < 38, then 1
            else:
                sub.append(2)
                #X[i][j] = 2 #if age >= 38, then 2
        elif j == 1: #checking workClass
            tempIndex = workingClass.index(db[i][j]) #checking index in workingClass
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1 #assigning correct index to X
        elif j == 2: #checking finalweight
            if int(db[i][j]) < 189778:
                sub.append(1)
                #X[i][j] = 1  # if fnlWeight < 189778, then 1
            else:
                sub.append(2)
                #X[i][j] = 2  # if fnlWeight >= 189778, then 2
        elif j == 3: #checking education
            tempIndex = education.index(db[i][j]) #checking index in education
            if int(tempIndex) < 9: # 9 = someCollege
                sub.append(1)
                #X[i][j] = 1  # if education < someCollege, then 1
            else:
                sub.append(2)
                #X[i][j] = 2  # if education >= someCollege, then 2
        elif j == 4: #checking education-num
            if int(db[i][j]) < 10: # 10 = someCollege
                sub.append(1)
                #X[i][j] = 1  # if education-num < someCollege, then 1
            else:
                sub.append(2)
                #X[i][j] = 2  # if education-num >= someCollege, then 2
        elif j == 5: #checking marital-status
            tempIndex = maritalStatus.index(db[i][j])  # checking index in maritalStatus
            sub.append(tempIndex+1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 6: #checking occupation
            tempIndex = occupation.index(db[i][j])  # checking index in occupation
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 7: #checking relationship
            tempIndex = relationship.index(db[i][j])  # checking index in relationship
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 8: #checking race
            tempIndex = race.index(db[i][j])  # checking index in race
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 9: #checking sex
            tempIndex = sex.index(db[i][j]) #checking index in sex
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 10: #checking capital-gain. min = 0, max = 99999, mean = 1077
            if int(db[i][j]) < 1077:  # if capital-gain is less than 1077
                sub.append(1)
                #X[i][j] = 1  # assign 1
            else:
                sub.append(2)
                #X[i][j] = 2  # else assign 2
        elif j == 11: #checking capital-loss. min = 0, max = 4356, mean = 87
            if int(db[i][j]) < 87:  # if capital-loss is less than 87
                sub.append(1)
                #X[i][j] = 1  # assign 1
            else:
                sub.append(2)
                #X[i][j] = 2  # else assign 2
        elif j == 12: #checking hours per week. min = 1, max = 99, mean = 40
            if int(db[i][j]) < 40:  # if hours are less than 40
                sub.append(1)
                #X[i][j] = 1  # assign 1
            else:
                sub.append(2)
                #X[i][j] = 2  # else assign 2
        elif j == 13: #checking native-country
            tempIndex = nativeCountry.index(db[i][j])  # checking index in country
            sub.append(tempIndex+1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 14: #checking class (income)
            tempIndex = income.index(db[i][j])  # checking index in maritalStatus
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
    X.append(sub)



#X = [[1, 1, 2, 2], [2, 1, 2, 1], [3, 1, 2, 2], [3, 1, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 2], [3, 1, 1, 2], [2, 2, 2, 2], [1, 1, 1, 2]]


#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
#Y = [2, 2, 2, 1, 1, 1, 2, 2, 2, 1]

Y.clear()
for sub_list in X:
    Y.append(sub_list.pop(-1))

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)
print("Classifier has been created...")

#plotting the decision tree
#tree.plot_tree(clf, feature_names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], class_names=['<=50K','>50K'], filled=True, rounded=True)
#print("Now showing tree...")
#plt.show()
#print("... tree has been printed!")

#reading the test data in a csv file
with open('adult.TEST', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      dbTest.append (row)
      #print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

#--> add your Python code here

for i in range(rowsTest):
    sub = []
    for j in range(columns):

        if j == 0: #checking age
            if int(dbTest[i][j]) < 38:
                sub.append(1)
                #X[i][j] = 1 #if age < 38, then 1
            else:
                sub.append(2)
                #X[i][j] = 2 #if age >= 38, then 2
        elif j == 1: #checking workClass
            tempIndex = workingClass.index(dbTest[i][j]) #checking index in workingClass
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1 #assigning correct index to X
        elif j == 2: #checking finalweight
            if int(dbTest[i][j]) < 189778:
                sub.append(1)
                #X[i][j] = 1  # if fnlWeight < 189778, then 1
            else:
                sub.append(2)
                #X[i][j] = 2  # if fnlWeight >= 189778, then 2
        elif j == 3: #checking education
            tempIndex = education.index(dbTest[i][j]) #checking index in education
            if int(tempIndex) < 9: # 9 = someCollege
                sub.append(1)
                #X[i][j] = 1  # if education < someCollege, then 1
            else:
                sub.append(2)
                #X[i][j] = 2  # if education >= someCollege, then 2
        elif j == 4: #checking education-num
            if int(dbTest[i][j]) < 10: # 10 = someCollege
                sub.append(1)
                #X[i][j] = 1  # if education-num < someCollege, then 1
            else:
                sub.append(2)
                #X[i][j] = 2  # if education-num >= someCollege, then 2
        elif j == 5: #checking marital-status
            tempIndex = maritalStatus.index(dbTest[i][j])  # checking index in maritalStatus
            sub.append(tempIndex+1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 6: #checking occupation
            tempIndex = occupation.index(dbTest[i][j])  # checking index in occupation
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 7: #checking relationship
            tempIndex = relationship.index(dbTest[i][j])  # checking index in relationship
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 8: #checking race
            tempIndex = race.index(dbTest[i][j])  # checking index in race
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 9: #checking sex
            tempIndex = sex.index(dbTest[i][j]) #checking index in sex
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 10: #checking capital-gain. min = 0, max = 99999, mean = 1077
            if int(dbTest[i][j]) < 1077:  # if capital-gain is less than 1077
                sub.append(1)
                #X[i][j] = 1  # assign 1
            else:
                sub.append(2)
                #X[i][j] = 2  # else assign 2
        elif j == 11: #checking capital-loss. min = 0, max = 4356, mean = 87
            if int(dbTest[i][j]) < 87:  # if capital-loss is less than 87
                sub.append(1)
                #X[i][j] = 1  # assign 1
            else:
                sub.append(2)
                #X[i][j] = 2  # else assign 2
        elif j == 12: #checking hours per week. min = 1, max = 99, mean = 40
            if int(dbTest[i][j]) < 40:  # if hours are less than 40
                sub.append(1)
                #X[i][j] = 1  # assign 1
            else:
                sub.append(2)
                #X[i][j] = 2  # else assign 2
        elif j == 13: #checking native-country
            tempIndex = nativeCountry.index(dbTest[i][j])  # checking index in country
            sub.append(tempIndex+1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
        elif j == 14: #checking class (income)
            tempIndex = income.index(dbTest[i][j])  # checking index in maritalStatus
            sub.append(tempIndex + 1)
            #X[i][j] = tempIndex + 1  # assigning correct index to X
    Xtest.append(sub)

#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
#Y = [2, 2, 2, 1, 1, 1, 2, 2, 2, 1]

Ytest.clear()
for sub_list in Xtest:
    Ytest.append(sub_list.pop(-1))

print("Test data is ready")

accuratePredictions = 0
totalPredictions = 0
iterator = 0

for data in dbTest:
    #transform the features of the test instances to numbers following the same strategy done during training,
    #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
    #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
    #--> add your Python code here

    class_predicted = clf.predict([Xtest[iterator]])[0]


    #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
    #--> add your Python code here

    if class_predicted == Ytest[iterator]:
        accuratePredictions += 1

    totalPredictions += 1
    iterator += 1

#find the lowest accuracy of this model during the 10 runs (training and test set)
#--> add your Python code here
currentAccuracy = accuratePredictions/totalPredictions

print("Total accuracy: ", currentAccuracy)