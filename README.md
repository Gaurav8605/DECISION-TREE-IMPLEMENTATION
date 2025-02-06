# DECISION-TREE-IMPLEMENTATION
COMPANY: CODETECH IT SOLUTIONS

NAME:Gaurav Baburao Gaikwad

INTER ID:CODHC18

DOMAIN: MACHINE LEARNING

DURATION:4 WEEKS

MENTOR:NEELA SANTHOSH KUMAR


#The provided Python code demonstrates the implementation of a Decision Tree Classifier on the famous Iris dataset using the scikit-learn library. The code covers essential machine learning tasks such as data loading, preprocessing, model training, evaluation, and visualization.

1. Importing Required Libraries
The first part of the code imports necessary Python libraries:

pandas for handling and manipulating structured data.
numpy for numerical computations.
matplotlib.pyplot and seaborn for data visualization.
sklearn for machine learning functionalities, including data splitting, model training, and evaluation.
2. Loading and Exploring the Dataset
The dataset is loaded using pd.read_csv('iris.csv'), assuming that the Iris dataset is available as a CSV file. The df.info() method provides an overview of the dataset, showing the number of entries, column names, and data types. Displaying df helps to check the structure of the dataset.

The dataset consists of four features:

Sepal Length
Sepal Width
Petal Length
Petal Width
These features define each flower instance, and the target variable (variety) represents different classes of the Iris species.
3. Splitting Data into Features and Target Variable
The feature columns are extracted into x, while the target labels (variety) are stored in y. The dataset is then split into training (80%) and testing (20%) sets using train_test_split(), ensuring that model training and evaluation occur on different data subsets.

4. Training the Decision Tree Classifier
A Decision Tree Classifier is created using DecisionTreeClassifier(criterion='entropy'). The entropy criterion is chosen as a measure to split nodes by calculating information gain. The classifier is trained using DT.fit(x_train, y_train), allowing it to learn patterns from the training data.

5. Making Predictions
Once trained, the model makes predictions on the test data using DT.predict(x_test). The predicted labels are printed to compare them with the actual labels.

6. Evaluating the Model
The performance of the classifier is evaluated using several metrics:

Accuracy Score: accuracy_score(y_test, pre) calculates the ratio of correct predictions to the total predictions.
Confusion Matrix: confusion_matrix(y_test, pre) provides a matrix representation of the classifier’s performance by comparing true labels to predicted labels.
Classification Report: classification_report(y_test, pre) generates a summary including precision, recall, and F1-score for each class.
7. Visualizing the Decision Tree
Finally, the trained decision tree is visualized using tree.plot_tree(DT, filled=True). This visualization provides insights into how the model makes decisions by showing feature splits at different nodes.

Conclusion
This code effectively demonstrates how to implement a Decision Tree Classifier on the Iris dataset. It follows a structured approach—data loading, preprocessing, model training, evaluation, and visualization—making it a comprehensive example for beginners in machine learning. The results offer insights into how decision trees classify different Iris species based on their features.
