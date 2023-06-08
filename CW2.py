import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import matplotlib.pyplot as plt

print("The Random Forest model is using...")
print()

# Dealing with Part A question
print("Analysing the Part A question")

# Load the training data
print("loading TrainingDataBinary.csv...")
train_data = pd.read_csv('G:\TrainingDataBinary.csv', header=None)

# Extract the features (eigenvalues) and labels from the training data
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Random Forest classifier
classifier_train_A = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the classifier on the training data
print("Training...")
classifier_train_A.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = classifier_train_A.predict(X_test)

# Print the predicted labels
print("The F1 score of the Random Forest model is: ")
print(f1_score(y_test, y_pred, average='macro'))
print("The accuracy of the Random Forest model is: ")
print(accuracy_score(y_test, y_pred))

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classifier_train_A.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier_train_A.classes_)
disp.plot()
plt.show()


# Dealing with Part B question
print()
print("Analysing the Part B question")

# Load the training data
print("loading TrainingDataMulti.csv...")
train_data = pd.read_csv('G:\TrainingDataMulti.csv', header=None)

# Extract the features (eigenvalues) and labels from the training data
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Random Forest classifier
classifier_train_B = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the classifier on the training data
print("Training...")
classifier_train_B.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = classifier_train_B.predict(X_test)

# Print the predicted labels
print("The F1 score of the Random Forest model is: ")
print(f1_score(y_test, y_pred, average='macro'))
print("The accuracy of the Random Forest model is: ")
print(accuracy_score(y_test, y_pred))

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classifier_train_B.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier_train_B.classes_)
disp.plot()
plt.show()

print("Test Part...The Random Forest model is using...")
print()

# Dealing with Part A question
print("Analysing the Part A question")

# Load the training data
print("loading TrainingDataBinary.csv...")
train_data = pd.read_csv('G:\TrainingDataBinary.csv', header=None)

# Extract the features (eigenvalues) and labels from the training data
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Create a Random Forest classifier
classifier_test_A = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the classifier on the training data
print("Training...")
classifier_test_A.fit(X, y)

# Load the testing data
print()
print("loading TestingDataBinary.csv...")
test_data = pd.read_csv('G:\TestingDataBinary.csv', header=None)

# Extract the features from the testing data
X_test = test_data.values

# Predict the labels for the testing data
y_pred = classifier_test_A.predict(X_test)

# Print the predicted labels
print("The predicted labels are:")
print(y_pred)

# Create a DataFrame with predicted labels and system traces
output_data = pd.DataFrame(data=X_test, columns=test_data.columns)
output_data['Label'] = y_pred

# Save the output DataFrame to a CSV file
output_data.to_csv('TestingResultsBinary.csv', index=False, header=False)

# Dealing with Part B question
print()
print("Analysing the Part B question")

# Load the training data
print("loading TrainingDataMulti.csv...")
train_data = pd.read_csv('G:\TrainingDataMulti.csv', header=None)

# Extract the features (eigenvalues) and labels from the training data
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Create a Random Forest classifier
classifier_test_B = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the classifier on the training data
print("Training...")
classifier_test_B.fit(X, y)

# Load the testing data
print()
print("loading TestingDataMulti.csv...")
test_data = pd.read_csv('G:\TestingDataMulti.csv', header=None)

# Extract the features from the testing data
X_test = test_data.values

# Predict the labels for the testing data
y_pred = classifier_test_B.predict(X_test)

# Print the predicted labels
print("The predicted labels are:")
print(y_pred)

# Create a DataFrame with predicted labels and system traces
output_data = pd.DataFrame(data=X_test, columns=test_data.columns)
output_data['Label'] = y_pred

# Save the output DataFrame to a CSV file
output_data.to_csv('TestingResultsMulti.csv', index=False, header=False)