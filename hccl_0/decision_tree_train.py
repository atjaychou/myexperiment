import torch
import torchvision
import yaml
from torchvision import transforms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz
from sklearn import tree

# 提取特征
from hccl.cifar100_emb_train import setup_seed, get_data_loaders
from hccl.model.classifiermodel import SimCLRClassifier


def extract_features(data_loader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, target in data_loader:
            outputs = model(images)
            features.extend(outputs.numpy())
            labels.extend(target.numpy())
    return features, labels

def classifier_model(checkpath, n_classes):
    freeze_base = True
    model = SimCLRClassifier(n_classes, freeze_base, checkpath)

    return model

if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(100)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    train_loader, test_loader, augmented_train_loader, _ = get_data_loaders(dataset)
    checkpath = config["checkpath"]
    # 0.001-0.0001
    classify_learning_rate = config["classify_learning_rate"]
    n_classes = config["n_classes_fine"]
    model = classifier_model(checkpath, n_classes)

    train_features, train_labels = extract_features(train_loader, model)
    test_features, test_labels = extract_features(test_loader, model)

    # 使用决策树模型进行训练和验证
    tree_model = DecisionTreeClassifier()
    tree_model.fit(train_features, train_labels)

    # 在测试集上进行预测和评估
    predictions = tree_model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # tree.plot_tree(tree_model)
    #
    # 使用Graphviz可视化决策树
    dot_data = tree.export_graphviz(tree_model, out_file=None,
                                    feature_names=[f"feature_{i}" for i in range(len(train_features[0]))])
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")  # 保存决策树图像为decision_tree.pdf或decision_tree.png
    graph.view()  # 在默认图形查看器中显示决策树