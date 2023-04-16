% Load the data (blood pressure and cholesterol measurements for 20 patients)
data = load('patient_data.txt');

% Normalize the data (mean 0, variance 1)
data_norm = zscore(data);

% Perform k-means clustering with 2 clusters
k = 2;
[clusters, centroids] = kmeans(data_norm, k);

% Plot the data points with different colors for each cluster
figure;
scatter(data_norm(:,1), data_norm(:,2), [], clusters, 'filled');
xlabel('Blood Pressure (mm Hg)');
ylabel('Cholesterol (mg/dL)');
title('Patient Clustering');

% Group the patients into high-risk and low-risk categories based on the clusters
low_risk = find(clusters == 1);
high_risk = find(clusters == 2);

% Print the results
fprintf('Low-risk patients: %s\n', mat2str(low_risk));
fprintf('High-risk patients: %s\n', mat2str(high_risk));

