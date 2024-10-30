# ML-Auction-verification

**1. Introduction**

The problem statement involves predicting the ‘verification.time’ for a process using machine
learning models. The verification.time is the time taken to verify a process, and it is shaped
by various features provided in the dataset. The dataset includes features related to process
capacities, property attributes, and the outcome of the verification result. The aim is to
develop and evaluate different regression models to accurately predict the verification.time
based on these features.

**2. Dataset and Features**

The dataset consists of 2043 samples and 9 features.
1. process.b1.capacity (int64) - Capacity measure for process batch 1.
2. process.b2.capacity (int64) - Capacity measure for process batch 2.
3. process.b3.capacity (int64) - Capacity measure for process batch 3.
4. process.b4.capacity (int64) - Capacity measure for process batch 4.
5. property.price (int64) - Price
6. property.product (int64) - Product attribute of property
7. property.winner (int64) - Indicates if property was a winning one.
8. verification.result (bool) - Result of the verification process (True/False).
9. verification.time (float64) - The time taken for the verification process.

**3. Methods**

Following are the various methods used in this project. Splitted the dataset into 75% training
and 25% testing.

In the notebook, feature scaling was performed using StandardScaler from the
sklearn.preprocessing module. StandardScaler standardizes features by removing the mean
and scaling to unit variance.

A loop is used to train and evaluate several regression models on a given dataset. A model
Dictionary is defined named models maps model names (as keys) to instances of various
regression models (as values). Each model is instantiated with appropriate parameters, such
as kernel type for SVR or random_state for reproducibility in tree-based models. An empty
dictionary named ‘scores’ is created to store the performance scores of each model after
evaluation.

_for name, model in models.items():
model.fit(X_train_scaled, y_train)
scores[name] = model.score(X_test_scaled, y_test)_

The for loop iterates over each key-value pair in the models dictionary. Each model is trained
on the scaled training data (X_train_scaled, y_train). The fit method adjusts the model
parameters to best fit the training data. The ‘score’ method computes the R^2 score
(coefficient of determination) of the model on the scaled test data (‘X_test_scaled, y_test’).
The R^2 score measures how well the model's predictions match the actual values. This
score is then stored in the ‘scores’ dictionary with the model name as the key.

**3.1 Baseline - Linear Regression**

It is a fundamental statistical method used for predicting a continuous dependent variable
(target) based on one or more independent variables (features). The basic idea is to fit a
linear equation to the observed data.
Result: R² = 0.3800

**3.2 Support Vector Machines**

Support Vector Machines (SVMs) are supervised learning algorithms used primarily for
classification and regression tasks. They work by finding the optimal hyperplane that
maximally separates different classes of data points in a high-dimensional space. SVMs
excel at handling complex, non-linear problems through the use of kernel functions, which
transform the input data into a higher-dimensional space where linear separation becomes
possible. The key strength of SVMs lies in their ability to maximise the margin—the distance
between the separating hyperplane and the nearest data points (called support vectors)—
which often results in better generalisation to unseen data. This makes SVMs particularly
effective for a wide range of applications, from text categorization and image classification to
bioinformatics and financial analysis, especially when dealing with high-dimensional data or
when the number of dimensions exceeds the number of samples.
SVR (Linear) SVR (Poly) SVR (RBF) | R² = -0.1927
| R² = -0.3035
| R² = -0.3004

**3.3 Decision Tree**

Decision Trees are intuitive supervised learning algorithms used for both classification and
regression tasks. They create a tree-like model of decisions based on features in the data,
splitting the dataset into increasingly homogeneous subsets. At each node, the algorithm
selects the most informative feature and threshold to maximize information gain or minimize
impurity. This process continues recursively, forming branches and leaves that represent
decision rules and outcomes. Decision Trees are valued for their interpretability, as the
resulting model can be easily visualized and explained. They can handle both numerical and
categorical data, making them versatile for various applications. However, they can be prone
to overfitting, especially when allowed to grow too deep. Despite this limitation, Decision
Trees remain popular in fields like finance, healthcare, and marketing due to their
transparency and ability to model complex decision processes. Perform experiment and
report the result obtained
Result: R² = 0.9889

**3.4 Random Forest**

Random Forest is an ensemble learning method that builds upon the concept of Decision
Trees. It creates multiple decision trees during training and combines their outputs to make
predictions, which helps to reduce overfitting and improve generalization. The "random"
aspect comes from two key features: each tree is trained on a random subset of the data
(bootstrap sampling), and at each node, only a random subset of features is considered for
splitting. This randomness introduces diversity among the trees, making the forest more
robust. The final prediction is typically the mode of the classes (for classification) or the
mean prediction (for regression) of the individual trees. Random Forests often achieve
higher accuracy than single decision trees and are less prone to overfitting. They can handle
high-dimensional data, are relatively fast to train, and can provide measures of feature
importance. These qualities make Random Forests popular in various domains, including
finance, healthcare, and image classification.
Result: R² = 0.9936

**3.5 AdaBoost**

AdaBoost, short for Adaptive Boosting, is an ensemble learning algorithm that combines
multiple weak learners to create a strong classifier. It works by iteratively training weak
classifiers (typically shallow decision trees) on the dataset, with each iteration focusing more
on the misclassified examples from previous rounds. AdaBoost assigns weights to training
samples, increasing weights for misclassified instances and decreasing them for correctly
classified ones. This adaptive approach allows the algorithm to focus on the harder-to-
classify examples over time. The final model is a weighted combination of these weak
learners, where better-performing classifiers are given more influence. AdaBoost is known
for its ability to achieve high accuracy, resist overfitting (to some extent), and handle various
types of data. It's particularly effective for binary classification problems and has been
successfully applied in areas such as face detection, medical diagnosis, and financial
prediction.
Result: R² = 0.9534

**3.6 Gradient Boosting**

Gradient Boosting is a powerful ensemble learning technique that builds a series of weak
learners, typically decision trees, to create a strong predictive model. It works by iteratively
adding new models to correct the errors made by existing models. Unlike AdaBoost, which
adjusts the weights of training samples, Gradient Boosting focuses on minimizing a loss
function by adding models that follow the negative gradient of this loss. Each new tree is
trained to predict the residuals (errors) of the previous ensemble of trees. The algorithm then
combines these trees additively, with each tree contributing a small improvement to the
overall model. Gradient Boosting is known for its high predictive accuracy and ability to
handle complex, non-linear relationships in data. It can be used for both regression and
classification tasks and has gained popularity in various domains, including web search
ranking, ecology, and financial forecasting. However, it can be prone to overfitting if not
properly tuned, and may require more computational resources compared to simpler
methods.
Result: R² = 0.9783

4. Results

● Top performers:
1. Random Forest and Decision Tree both achieved the highest score of 0.99
(R²).
2. Gradient Boosting follows closely with a score of 0.98.
3. AdaBoost also performed well with a score of 0.95.
● Moderate performer:
Linear Regression achieved a score of 0.38.
● Poor performers:
1. SVR (Support Vector Regression) models all showed negative R² scores:
SVR (Linear): -0.19
SVR (Poly): -0.30
SVR (RBF): -0.30
Tree-based ensemble methods (Random Forest, Decision Tree, Gradient Boosting, and
AdaBoost) performed exceptionally well on the test data. Linear Regression showed
moderate performance, while the SVR models performed poorly, indicating they may not be
suitable for the particular dataset or problem.

**5. Hyperparameter Tuning**

**5.1 SVM with RBF Kernel**

Used grid search to optimize two parameters: C and Gamma. The parameter C controls the balance between achieving low training error and minimizing the overall error. Gamma determines the extent of influence a single training example has, with low values indicating a broader influence and high values indicating a more localized influence.
Best parameters:
C: 9
Gamma: 0.1
Best Grid Search Score: -0.2404809564950594
Result after the 25% splitting: 0.1590114060994929

** 5.2 Decision Trees**

Hyper-parameters of a Decision Tree:
Max depth: Helps prevent the depth of the tree from becoming too deep and controls overfitting.
Maximum features: Number of features to consider when looking for the best fit.
Best parameters:
Max depth: 18
Max features: 7
Random state: 42
Best Grid Search Score: 5
Result after splitting: 0.9855897350665728

**5.3 Random Forest**

n_estimators: Number of trees in the forest
Parameter grid:
Grid search:
Max depth: 0 to 9
n_estimators: 7
Best Grid Search Score: 0.9902158240331055
Result after splitting: 0.99417930631355

**5.4 AdaBoost**

Learning rate: Weight applied to each classifier at each boosting iteration
n_estimators: Number of the boosting stages for the running
Parameter grid:
Grid search:
Learning rate: A range from 0.1 to 1 in steps of 0.1 (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
n_estimators: A range from 10 to 100 in steps of 10 (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
Results Obtained for the Best Configuration:
Best Parameters:
Learning rate: 1.0
n_estimators: 50
Best Grid Search Score: 
Result for the 25% testing dataset:
Grid Search Score: 0.9533601385491626
AdaBoost Score: 0.9469427142538824

**5.5 Gradient Boosting**

Hyper-parameters of a Gradient Boosting:
Parameter grid:
Grid search:
Max depth: A range from 1 to 10 (1, 2, 3, 4, 5, 6, 7, 8, 9)
n_estimators: 'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]

Results Obtained for the Best Configuration:
Best Parameters:
Max depth: 7
n_estimators: 91
Best Grid Search Score: 0.9782705398483936
Result for the 25% testing dataset:
Gradient Boosting Score: 0.9978172141865338
