# knee-xray-classification
This repository contains the code and models developed for my Master's thesis titled:

**"Classification of Knee Bone Health Status Using Deep Learning: Normal, Osteopenia, and Osteoporosis Detection from X-ray Images"**

##  Problem Statement

Early detection of bone density issues like osteopenia and osteoporosis is crucial for preventing disability. This project aims to classify knee X-ray images into three clinically important categories — **Normal**, **Osteopenia**, and **Osteoporosis** — using deep learning.The main research question guiding this study is: How does the classification performance compare across different deep learning models, and can it be improved using ensemble learning for classifying knee X-ray images into normal, osteopenia, and osteoporosis categories? This main question is further explored through the following sub-questions:
How does the performance of an ensemble model, combining predictions from multiple deep learning architectures including a custom CNN and transfer learning models, compare to individual models in this classification task?
How do different ensemble fusion strategies (e.g., stacking, majority voting, weighted ensemble, maximum voting, and average ensemble) influence the overall classification performance in distinguishing normal, osteopenia, and osteoporosis cases?


##  Methods

I implemented and compared the following models:
- ✅ Custom Convolutional Neural Network (CNN)
- ✅ Pretrained models: ResNet50 and DenseNet121
- ✅ Ensemble learning strategies: Average, Weighted, Maximum Voting, Majority Voting, and Stacking

##  Dataset

- Source: Publicly available Kaggle dataset
- Total images: 5,200
- Split: 70% Training | 20% Validation | 10% Test
- Three classes: `Normal`, `Osteopenia`, `Osteoporosis`

> Note: Due to size restrictions, dataset files are not included. You can download them from https://www.kaggle.com/datasets/fuyadhasanbhoyan/knee-osteoarthritis-classification-224224/data.

## 📈 Results

| Model              | Test Accuracy | Macro AUC |
|-------------------|---------------|------------|
| Custom CNN         | 79.6%         | 0.90       |
| ResNet50           | 87.6%         | 0.96       |
| DenseNet121        | 87.8%         | 0.97       |
| Stacking Ensemble  | 89.3%         | 0.97       |

