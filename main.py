# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

X_training = []
y_training = []
dbTraining = []
dbTest = []
num_bootstraps = 20

# flags for easy combination testing
impute = True
dropNa = False
upSamp = False
downSamp = False

# imputer for missing values
preprocessor = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# read training data
df = pd.read_csv('adult.data', sep=',', header=0, na_values='?')

# init dbTraining according to flags
if impute:
    dbTraining = preprocessor.fit_transform(df.values)
elif dropNa:
    dbTraining = np.array(df.dropna().values)[:, :]
else:
    dbTraining = np.array(df.values)[:, :]

# majority and minority classes for re-sampling
minority_class = [row for row in dbTraining if row[-1] == '>50K']
majority_class = [row for row in dbTraining if row[-1] == '<=50K']

# read test data
df = pd.read_csv('adult.test', sep=',', header=0, na_values='?')

# init dbTest according to flags
if impute:
    dbTest = preprocessor.fit_transform(df.values)
elif dropNa:
    dbTest = np.array(df.dropna().values)[:, :]
else:
    dbTest = np.array(df.values)[:, :]

# init class votes
classVotes = [[0, 0] for i in range(len(dbTest))]

# dictionaries to convert data to numeric values
workclass = {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7, np.nan: 8}
education = {'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5, 'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12, 'Doctorate': 13, '5th-6th': 14, 'Preschool': 15, np.nan: 16}
marital = {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2, 'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6, np.nan: 7}
occupation = {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12, 'Armed-Forces': 13, np.nan: 14}
relationship = {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5, np.nan: 6}
race = {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4, np.nan: 5}
sex = {'Female': 0, 'Male': 1, np.nan: 2}
country = {'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5, 'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11,  'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18, 'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24, 'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31, 'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36, 'Trinadad&Tobago': 37, 'Peru': 38, 'Hong': 39, 'Holand-Netherlands': 40, np.nan: 41}
income = {'>50K': 0, '<=50K': 1}
cont = {np.nan: -1}

# accuracy of single classifier
single_correct = 0

print("Started my base and ensemble classifier ...")

# we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample
for k in range(num_bootstraps):

    bootstrapSample = []
    X_training = []
    y_training = []

    # create new bootstrap sample for each classifier according to classifier
    if upSamp:
        up_sample = resample(minority_class, replace=True, n_samples=len(majority_class))
        combined_sample = up_sample + majority_class
        bootstrapSample = resample(combined_sample, n_samples=len(dbTraining), replace=True)
    elif downSamp:
        down_sample = resample(majority_class, replace=False, n_samples=len(minority_class))
        combined_sample = down_sample + minority_class
        bootstrapSample = resample(combined_sample, n_samples=len(dbTraining), replace=True)
    else:
        bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    # populate the values of X_training and y_training by using the current bootstrap sample
    for sample in bootstrapSample:
        if impute is False and dropNa is False:
            X_training.append([sample[0], workclass[sample[1]], sample[2], education[sample[3]], sample[4], marital[sample[5]], occupation[sample[6]], relationship[sample[7]], race[sample[8]], sex[sample[9]], sample[10], sample[11], sample[12], country[sample[13]]])
        else:
            X_training.append([cont.get(sample[0], sample[0]), workclass[sample[1]], cont.get(sample[2], sample[2]), education[sample[3]], cont.get(sample[4], sample[4]), marital[sample[5]], occupation[sample[6]], relationship[sample[7]], race[sample[8]], sex[sample[9]], cont.get(sample[10], sample[10]), cont.get(sample[11], sample[11]), cont.get(sample[12], sample[12]), country[sample[13]]])
        y_training.append(income[sample[-1]])

    # fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
    clf = clf.fit(X_training, y_training)

    for i, testSample in enumerate(dbTest):

        # make classifier prediction and increment the class vote for said prediction
        class_predicted = clf.predict([[testSample[0], workclass[testSample[1]], testSample[2], education[testSample[3]], testSample[4], marital[testSample[5]], occupation[testSample[6]], relationship[testSample[7]], race[testSample[8]], sex[testSample[9]], testSample[10], testSample[11], testSample[12], country[testSample[13]]]])[0]
        classVotes[i][class_predicted] += 1

        # for only the first base classifier,
        # compare the prediction with the true label of the test sample here to start calculating its accuracy
        if k == 0 and int(class_predicted) == int(income[testSample[-1]]):
            single_correct += 1

    # for only the first base classifier, print its accuracy here
    if k == 0:
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print(f"My base classifier accuracy: {single_correct/len(dbTest)}")
        print()

# calculate the accuracy of the ensemble classifier
ensemble_correct = 0
for i, testSample in enumerate(dbTest):
    # check test class against index of max value in votes array
    if classVotes[i].index(max(classVotes[i])) == int(income[testSample[-1]]):
        ensemble_correct += 1

# printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print(f"My ensemble accuracy: {ensemble_correct/len(dbTest)}")
print()

print("Started Random Forest algorithm ...")

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=20)

# Fit Random Forest to the training data
clf.fit(X_training, y_training)

# test the accuracy of the random forest classifier
rf_correct = 0
for i, testSample in enumerate(dbTest):
    class_predicted = clf.predict([[testSample[0], workclass[testSample[1]], testSample[2], education[testSample[3]], testSample[4], marital[testSample[5]], occupation[testSample[6]], relationship[testSample[7]], race[testSample[8]], sex[testSample[9]], testSample[10], testSample[11], testSample[12], country[testSample[13]]]])[0]

    if int(class_predicted) == int(income[testSample[-1]]):
        rf_correct += 1

# printing Random Forest accuracy here
print(f"Random Forest accuracy: {rf_correct/len(dbTest)}")

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
