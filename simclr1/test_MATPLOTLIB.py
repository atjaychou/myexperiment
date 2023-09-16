import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston
from sklearn import tree

# 加载数据集
iris = load_iris()
# 准备数据
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 拟合
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
tree.plot_tree(clf,
               feature_names = iris.feature_names,
               class_names=iris.target_names,
               rounded=True,
               filled = True)
