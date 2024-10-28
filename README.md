

# Heart Disease Risk Analysis

## Project Overview

This Project is to analyze a dataset of medical records to identify patients at high risk of developing heart disease. This project involves using unsupervised learning techniques to cluster patients based on their medical history and identifying risk factors associated with heart disease.

## Dataset

The analysis uses the heart disease dataset from the UCI Machine Learning Repository. It consists of 303 instances and 14 attributes, including age, sex, chest pain type, blood pressure, serum cholesterol, and a target variable indicating the presence or absence of heart disease.

## Analysis Steps

The project includes the following key steps:

1. **Exploratory Data Analysis (EDA)**
   Load the dataset into a pandas DataFrame and perform EDA to gain insights.
   ```python
   dir = 'processed.cleveland.data'
   column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
   data = pd.read_csv(dir, header=None, names=column_names, na_values='?')
   print(data.head())
   data.info()
   ```

2. **Data Preprocessing**
   Handle missing values, scale features, and encode categorical variables.
   ```python
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import Pipeline
   X = dfe.drop("num", axis=1)
    y = dfe["num"]
    
    columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
   ```

3. **Clustering Techniques**
   Apply K-means clustering, hierarchical clustering, and DBSCAN clustering.
   ```python
   from sklearn.cluster import KMeans, AgglomerativeClustering
   from sklearn.metrics import silhouette_score, davies_bouldin_score
    resulst = {}
    unique_labels = np.unique(kmeans_labels)
    if len(kmeans_labels) > 1:
      silhouette_avg = silhouette_score(X_processed, kmeans_labels)
      davies_bouldin_avg = davies_bouldin_score(X_processed, kmeans_labels)
    
      results = {
          "silhouette_score": silhouette_avg,
          "davies_bouldin_score": davies_bouldin_avg
      }
    else:
      results = {
          "silhouette_score": None,
          "davies_bouldin_score": None
      }
   ```

4. **Visualization and Insight**
   Utilize PCA and t-SNE for cluster visualization.

5. **Risk Factor Identification**
   Employ Gaussian Mixture Models (GMMs) to identify risk factors.

6. **Performance Evaluation**
   Evaluate clusters using silhouette score and Davies-Bouldin index.

7. **Comparison and Conclusion**
   Compare the performance of clustering algorithms and select the best-performing model.

## Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn

## Repository Contents

- `README.md`: Project overview and analysis steps.
- `heart_disease_analysis.ipynb`: Jupyter notebook containing all code, data analysis, and visualizations.

