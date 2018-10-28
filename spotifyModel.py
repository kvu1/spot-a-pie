'''
Author: Kyle Vu
Date: 7/22/2018
Purpose: Create models to predict user track response
'''
# import necessary packages
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# read in data
spotify = pd.read_csv("/Users/kylevu/Desktop/spotifyProject/bigPlaylist.csv")
x = spotify.iloc[:, 3:16]
y = spotify.iloc[:, 16].values

# partition dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1618)

# scale features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


### LOGISTIC REGRESSION ###
# fit a logistic regression model
log_classifier = LogisticRegressionCV(cv = 10, random_state = 1618) # 10-fold cv
log_classifier.fit(x_train, y_train)

# predict test set results with logistic regressor
y_pred_log = log_classifier.predict(x_test)

# construct confusion matrix
# 140/226 predictions correct; more false positives than false negatives
confuse_log = confusion_matrix(y_test, y_pred_log)
confuse_frame_log = pd.DataFrame(confuse_log)
palette = sn.cubehelix_palette(8, as_cmap = True)
ax = plt.axes()
sn.heatmap(confuse_frame_log, annot = True, fmt = 'g', cmap = palette, ax = ax)
ax.set_title("Logistic Regression")
plt.xlabel('Predicted')
plt.ylabel('Observed')


### K-NEAREST NEIGHBORS ###
neighbors_lst = list(range(1, 101)) # creating list of K neighbors to test
knn_cv_scores = []
for k in neighbors_lst:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, x_train, y_train, cv = 10, scoring = 'accuracy') # 10-fold cv
    knn_cv_scores.append(scores.mean())

# find number of neighbors that optimizes MSE
knn_MSE = [1 - x for x in knn_cv_scores]
min_error = knn_MSE[0]
for i in range(len(knn_MSE)):
    min_error = min_error
    if knn_MSE[i] < min_error:
        knn_min_index = i
        min_error = knn_MSE[i]

# plotting accuracy rate
plt.plot(neighbors_lst, knn_MSE, '#2E4057')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.title("Tuning the KNN Model")
plt.show() # Misclassification optimized with 77 neighbors

# fit k-nearest neighbors model on training data
knn_classifier = KNeighborsClassifier(n_neighbors = knn_min_index + 1)
knn_classifier.fit(x_train, y_train)

# predict test set results with knn classifier
y_pred_knn = knn_classifier.predict(x_test)

# build confusion matrix
# 134/226 correct; more false negatives than false positives
confuse_knn = confusion_matrix(y_test, y_pred_knn)
confuse_frame_knn = pd.DataFrame(confuse_knn)
ax = plt.axes()
sn.heatmap(confuse_frame_knn, annot = True, fmt = 'g', cmap = palette, ax = ax)
ax.set_title('KNN')
plt.xlabel('Predicted')
plt.ylabel('Observed')


### DECISION TREE ###
trees_lst = list(range(1, 101))
tree_cv_scores = []
for max_depth in trees_lst:
    oak = DecisionTreeClassifier(max_depth = max_depth, random_state = 1618)
    scores = cross_val_score(oak, x_train, y_train, cv = 10, scoring = 'accuracy') # 10-fold cv
    tree_cv_scores.append(scores.mean())

# find value where MSE is optimized
tree_MSE = [1 - x for x in tree_cv_scores]
min_error = tree_MSE[0]
for i in range(len(tree_MSE)):
    min_error = min_error
    if tree_MSE[i] < min_error:
        tree_min_index = i
        min_error = tree_MSE[i]

# plotting acc. rate
plt.plot(trees_lst, tree_MSE, '#53917E')
plt.xlabel('Max Depth')
plt.ylabel('Misclassification Error')
plt.title("Tuning the Decision Tree Model")
plt.show() # Misclassification optimized with max depth of 3

# fit decision tree classifier
tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1618, max_depth = tree_min_index + 1)
tree_classifier.fit(x_train, y_train)

# predict test set results
y_pred_tree = tree_classifier.predict(x_test)

# build confusion matrix
# 141/226 correct; most errors are Type 2
confuse_tree = confusion_matrix(y_test, y_pred_tree)
confuse_frame_tree = pd.DataFrame(confuse_tree)
ax = plt.axes()
sn.heatmap(confuse_frame_tree, annot = True, fmt = 'g', cmap = palette, ax = ax)
ax.set_title('Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Observed')


### RANDOM FOREST ###
vote_lst = list(range(1, 101))
rf_cv_scores = []
for tree_count in vote_lst:
    maple = RandomForestClassifier(n_estimators = tree_count, random_state = 1618)
    scores = cross_val_score(maple, x_train, y_train, cv = 10, scoring = 'accuracy') # 10-fold cv
    rf_cv_scores.append(scores.mean())

# find minimum error's index (i.e. optimal num. of estimators)
rf_MSE = [1 - x for x in rf_cv_scores]
min_error = rf_MSE[0]
for i in range(len(rf_MSE)):
    min_error = min_error
    if rf_MSE[i] < min_error:
        rf_min_index = i
        min_error = rf_MSE[i]

# plotting acc. rate
plt.plot(vote_lst, rf_MSE, '#DA4167')
plt.xlabel('Number of Estimators')
plt.ylabel('Misclassification Error')
plt.title("Tuning the Random Forest Model")
plt.show() # Misclassification optimized with 62 estimators

# fit random forest classifier
forest_classifier = RandomForestClassifier(n_estimators = rf_min_index + 1, criterion = 'entropy', random_state = 1618)
forest_classifier.fit(x_train, y_train)

# predict test set
y_pred_forest = forest_classifier.predict(x_test)

# build confusion matrix
# 149/226 correct; more type 1 errors than type 2
confuse_forest = confusion_matrix(y_test, y_pred_forest)
confuse_frame_forest = pd.DataFrame(confuse_forest)
ax = plt.axes()
sn.heatmap(confuse_frame_forest, annot = True, fmt = 'g', cmap = palette, ax = ax)
ax.set_title("Random Forest")
plt.xlabel('Predicted')
plt.ylabel('Observed')

### GRID LAYOUT: HYPERPARAMETER TUNING ###
f, axarr = plt.subplots(3, sharex = True)
axarr[0].plot(neighbors_lst, knn_MSE, '#2E4057')
axarr[0].set(title = "KNN", xlabel = "Number of Neighbors")
axarr[1].plot(trees_lst, tree_MSE, '#53917E')
axarr[1].set(title = "Decision Tree", xlabel = 'Max Depth', ylabel = 'Misclassification Error')
axarr[2].plot(vote_lst, rf_MSE, '#DA4167')
axarr[2].set(title = "Random Forest", xlabel = 'Number of Estimators')
plt.subplots_adjust(hspace = 0.75) # add some vertical buffer between the three subplots