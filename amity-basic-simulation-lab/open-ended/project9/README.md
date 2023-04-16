# K-Means Clustering for Patient Risk Categorization

Given the data about Blood Pressure and Cholesterol for 20 patients, using k-means clustering to group the patients into having high risk of heart attack and those having low risk of heart attack.

|Column 1|Column 2|
| :- | :- |
|Blood Pressure|Cholesterol|

## Problem Definition

The aim of this project is to categorize patients into high-risk and low-risk categories based on their blood pressure and cholesterol levels using K-means clustering. The project aims to aid healthcare professionals in identifying patients who are at a higher risk of heart attacks.

## Problem Scope

The project is focused on using K-means clustering to group the patients into high-risk and low-risk categories. It does not include any further analysis or treatment recommendations.

## Technical Review

The project uses the K-means clustering algorithm to group patients based on their blood pressure and cholesterol levels. The algorithm iteratively assigns each data point to a cluster by minimizing the sum of squares between the data point and the assigned centroid. The optimal number of clusters is determined using the elbow method. The data is normalized before applying the clustering algorithm.

## Design Requirements

- Access to patient data containing blood pressure and cholesterol measurements
- Knowledge of the K-means clustering algorithm
- Familiarity with MATLAB programming language

## Design Description

### Overview

The patient data is loaded into the MATLAB environment and normalized using z-score. K-means clustering with two clusters is performed on the normalized data. The data points are plotted with different colors representing each cluster. Patients are grouped into high-risk and low-risk categories based on the clusters.

### Detailed Description

1. Load the patient data
2. Normalize the data
3. Perform K-means clustering with 2 clusters
4. Plot the data points with different colors for each cluster
5. Group the patients into high-risk and low-risk categories based on the clusters

### Use

1. Open the MATLAB environment
2. Load the patient data into the workspace
3. Copy and paste the code into the MATLAB command window
4. Run the code to perform the K-means clustering and patient categorization

## Evaluation

### Overview

The project evaluates the K-means clustering algorithm's ability to categorize patients into high-risk and low-risk categories based on their blood pressure and cholesterol levels.

### Prototype

The code in this project is the prototype for patient risk categorization using K-means clustering.

### Testing and Results

The code was tested using sample patient data containing blood pressure and cholesterol measurements for 20 patients. The K-means clustering algorithm was able to categorize patients into high-risk and low-risk categories with reasonable accuracy. The results were visually displayed in a scatter plot.

### Assessment

The project demonstrated that K-means clustering can be used to categorize patients into high-risk and low-risk categories based on their blood pressure and cholesterol levels. The results were consistent with the medical literature's recommendations on identifying patients at high risk of heart attacks.

### Next Steps

The project can be expanded to include more data variables, such as age, family history, and lifestyle factors, to improve the accuracy of patient categorization. The project can also be used to develop a decision support system for healthcare professionals to aid in the identification and management of patients at high risk of heart attacks.
