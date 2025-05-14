# Social Media Usage and Emotional Well-being: A Machine Learning Approach

## Abstract
This project explores the relationship between social media usage and emotional well-being using machine learning techniques. We analyze user data to uncover patterns, preprocess the data for modeling, and evaluate various machine learning models to predict dominant emotions. The study leverages advanced techniques such as feature scaling, dimensionality reduction, and ensemble learning to achieve robust results.

---

## 1. Introduction
Social media has become an integral part of modern life, influencing emotional well-being in profound ways. This project aims to analyze social media usage data to predict users' dominant emotions. By leveraging machine learning models, we aim to uncover insights into how social media impacts emotional states.

---

## 2. Methodology

### 2.1 Data Collection
The dataset consists of three files: `train.csv`, `test.csv`, and `val.csv`, containing user demographics, social media usage patterns, and emotional states.

### 2.2 Data Preprocessing
- **Handling Missing Values**: Rows with missing values were dropped to ensure data integrity.
- **Encoding Categorical Variables**: One-hot encoding was applied to variables such as `Gender` and `Platform`.
- **Feature Scaling**: StandardScaler was used to standardize numerical features.
- **Dimensionality Reduction**: PCA was employed to reduce feature dimensions while retaining variance.
- **Feature Selection**: Recursive Feature Elimination (RFE) was used to select the most relevant features.

### 2.3 Machine Learning Models
We evaluated the following models:
- Linear Regression
- Ridge and Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Regression
- K-Nearest Neighbors
- Extra Trees
- Ensemble Models (Voting and Stacking Regressors)

---

## 3. Results and Analysis

### 3.1 Exploratory Data Analysis (EDA)
- **Age Distribution**: Visualized using histograms.
- **Gender Distribution**: Analyzed using count plots.
- **Daily Usage Time**: Bar plots revealed platform-specific usage patterns.
- **Dominant Emotion Distribution**: Count plots highlighted prevalent emotional states.

### 3.2 Model Evaluation
- Metrics: RMSE, R², and pseudo-accuracy were used to evaluate model performance.
- Ensemble models, particularly the Stacking Regressor, achieved the highest accuracy and R² scores.

### 3.3 Visualization
- Confusion matrices and ROC curves were generated for classification models.
- Precision-recall curves provided additional insights into model performance.

---

## 4. Conclusion
This study demonstrates the potential of machine learning in analyzing social media's impact on emotional well-being. Ensemble models proved to be the most effective in predicting dominant emotions. Future work could involve incorporating additional features, such as sentiment analysis of social media posts, to enhance predictive accuracy.

---

## 5. References
1. Scikit-learn Documentation: https://scikit-learn.org/
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Seaborn Documentation: https://seaborn.pydata.org/
4. Matplotlib Documentation: https://matplotlib.org/

---

## 6. Acknowledgments
We thank the contributors of the dataset and the open-source community for providing the tools and libraries used in this project.

---

## 7. Code and Dataset
The complete code and dataset are available in the repository. Follow the instructions in the `socialmediaemotion-ml.ipynb` file to reproduce the results.