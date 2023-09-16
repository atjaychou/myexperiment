# # 数据预处理和增强
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # 创建数据集实例
# dataset = VOCClassificationDataset(dataset_dir, transform=transform)
# testData = VOCTestDataset(dataset_dir, transform=transform)
#
# # 创建数据加载器
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # 加载预训练的ResNet-18模型
# model = resnet18(pretrained=True)
# num_classes = len(class_names)
#
# # 替换最后一层全连接层
# model.fc = nn.Linear(model.fc.in_features, num_classes)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# # 示例：训练模型
# num_epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for images, labels, coarse_labels in dataloader:
#         images = images.to(device)
#         labels = labels.to(device)
#         coarse_labels = coarse_labels.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * images.size(0)
#
#     epoch_loss = running_loss / len(dataset)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#
# # 示例：使用训练好的模型进行预测
# model.eval()
#
# # 随机选择一张测试图像进行预测
# test_image_path = 'path/to/test/image.jpg'
# test_image = Image.open(test_image_path).convert('RGB')
# test_image = transform(test_image).unsqueeze(0).to(device)
#
# with torch.no_grad():
#     outputs = model(test_image)
#     _, predicted = torch.max(outputs, 1)
#     predicted_label = class_names[predicted.item()]
#     print("Predicted Label:", predicted_label)
#
# # 预测粗粒度标签
# coarse_outputs = model.fc(test_image)
# _, coarse_predicted = torch.max(coarse_outputs, 1)
# coarse_predicted_label = coarse_class_names[coarse_predicted.item()]
# print("Predicted Coarse Label:", coarse_predicted_label)
# print(111)
