# Data-Mining-II

1. K-Means Clustering: Geographic Analysis of Medical Charges
Objective: Identify regional patterns in additional medical charges using K-means clustering.
Key Steps:

Cleaned and scaled geographic (Lat/Lng) and charge data.

Applied K-means to group locations into 3 clusters based on charge similarity.

Insight: Clusters revealed high-charge regions (avg. 
23.7
K
)
,
m
o
d
e
r
a
t
e
(
23.7K),moderate(10.7K), and low ($9.9K), aiding targeted resource allocation.
Tools: Python (Pandas, Scikit-learn), Elbow Method, Seaborn.
Business Impact: Supports dynamic pricing and resource optimization for healthcare providers.

2. PCA for Medical Data Dimensionality Reduction
Objective: Simplify a medical dataset by retaining key features with PCA.
Key Steps:

Standardized 7 continuous variables (e.g., Income, VitD levels).

Identified 4 principal components explaining 71% of variance via Kaiser criterion.

Insight: Reduced dataset complexity while preserving critical trends for further analysis.
Tools: Python (Scikit-learn), Scree Plot, StandardScaler.
Business Impact: Enables efficient analysis of high-dimensional healthcare data.

3. Market Basket Analysis: Prescription Pattern Mining
Objective: Discover frequently co-prescribed medications using Apriori algorithm.
Key Steps:

Processed 7,501 prescriptions into transaction format.

Generated association rules, highlighting top pairs like metformin → abilify (confidence: 45.6%, lift: 1.9).

Insight: Common combinations (e.g., carvedilol + lisinopril) suggest opportunities for bundled treatments.
Tools: Python (MLxtend), Support/Confidence/Lift metrics.
Business Impact: Optimizes pharmacy inventory and personalized treatment plans.
