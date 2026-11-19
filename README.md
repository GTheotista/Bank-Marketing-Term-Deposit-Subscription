# Bank Marketing Term Deposit Subscription Prediction

**Tools:** Python, Scikit-learn, PyTorch  
**Visualizations:** Matplotlib, Seaborn  
**Dataset:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank)

---

## I. Business Problem Understanding

Bank marketing campaigns aim to identify customers who are likely to subscribe to term deposit products. However, contacting customers who are unlikely to subscribe can increase campaign costs and reduce marketing efficiency.

Predicting customer subscription likelihood before a marketing call enables banks to prioritize potential customers, allocate outreach resources more effectively, and improve campaign targeting strategies.

### Problem Statement

- **Customer Perspective:** Customers receive marketing calls offering term deposit products. Understanding customer characteristics and previous campaign interactions can help banks develop more relevant and targeted marketing strategies.
- **Institutional Perspective:** Banks need a predictive model to identify customers with a higher likelihood of subscribing to term deposits. This helps marketing teams prioritize outreach activities and optimize limited campaign resources.

### Goals

1. Develop a machine learning and deep learning model to predict whether a customer will subscribe to a bank term deposit.
2. Compare Logistic Regression, Random Forest, and Multilayer Perceptron (MLP) performance using appropriate evaluation metrics.
3. Identify influential customer, campaign, and economic features using Permutation Feature Importance and SHAP.
4. Provide actionable insights to support customer prioritization and improve marketing campaign efficiency.

### Business Impact

- Enables marketing teams to prioritize customers with higher predicted subscription likelihood.
- Improves the allocation of limited campaign outreach resources.
- Supports data-driven decisions regarding campaign timing and communication channels.
- Helps identify important customer and economic patterns associated with term deposit subscriptions.
- Provides a foundation for developing a predictive customer targeting system.

---

## II. Data Understanding & Preprocessing

### Dataset Source

[UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank)

The project uses the `bank-additional-full.csv` dataset, which contains 41,188 observations and 21 original columns, including customer information, campaign-related variables, economic indicators, and the target variable `y`.

### Key Steps

- **Data Inspection & Cleaning:** Examined dataset structure, data types, missing values, duplicate observations, and categorical values.
- **Target Analysis:** Identified an imbalanced target distribution, with approximately 88.74% `no` and 11.26% `yes` subscription outcomes.
- **Feature Engineering:** Transformed `pdays` into `previously_contacted` and `days_since_previous_contact` to represent previous contact history more meaningfully.
- **Data Leakage Prevention:** Excluded `duration` because it is only available after a marketing call has taken place and would introduce target leakage in a pre-call prediction scenario.
- **Feature Selection:** Selected 20 predictor features consisting of 10 numerical and 10 categorical features.
- **Outlier Handling:** Retained extreme observations and applied RobustScaler to numerical features to reduce sensitivity to outliers.
- **Numerical Preprocessing:** Applied median imputation and RobustScaler to numerical variables.
- **Categorical Preprocessing:** Applied most frequent imputation and One-Hot Encoding with `handle_unknown="ignore"`.
- **Pipeline Integration:** Combined numerical and categorical preprocessing using ColumnTransformer to maintain consistent transformations and prevent data leakage during model training.
- **Data Splitting:** Applied an 80:20 stratified train-test split to preserve the target class distribution.

---

## III. Modeling

### Algorithms Tested

- Logistic Regression
- Random Forest
- Multilayer Perceptron (MLP) using PyTorch

### Model Development

- Logistic Regression and Random Forest were implemented as classical machine learning approaches.
- Both machine learning models used class weighting to address target class imbalance.
- MLP was developed as a deep learning model with two hidden layers containing 128 and 64 neurons.
- ReLU activation, Batch Normalization, and Dropout were applied within the MLP architecture.
- Adam optimizer and BCEWithLogitsLoss were used for model training.
- A Stratified 5-Fold Cross-Validation strategy was used for model comparison.
- PR-AUC (Average Precision) was selected as the primary model selection metric, with ROC-AUC as a supporting metric.
- Classification thresholds were determined using out-of-fold predictions for machine learning models and an internal validation procedure for MLP.

### Best Model

- **MLP (PyTorch Multilayer Perceptron)**

MLP achieved the highest mean cross-validation PR-AUC of 0.448, followed by Logistic Regression at 0.444 and Random Forest at 0.372.

### Performance (Test Set)

| Metric | MLP | Logistic Regression | Random Forest |
|---|---:|---:|---:|
| Precision@100 | 0.800 | 0.750 | 0.510 |
| Recall@100 | 0.086 | 0.081 | 0.055 |
| Lift@100 | 7.102 | 6.658 | 4.527 |
| F1-score | 0.486 | 0.485 | 0.467 |
| PR-AUC | 0.461 | 0.457 | 0.377 |
| ROC-AUC | 0.795 | 0.798 | 0.776 |

MLP achieved the highest Precision@100, Lift@100, F1-score, and PR-AUC. Logistic Regression achieved a slightly higher ROC-AUC of 0.798 compared with 0.795 for MLP.

### Best Model Evaluation

The final MLP model was evaluated using an optimized classification threshold of approximately 0.695.

- Accuracy: 0.8745
- Precision: 0.4510
- Recall: 0.5259
- F1-score: 0.4856
- True Positives: 488
- True Negatives: 6,716
- False Positives: 594
- False Negatives: 440

For the positive class (`yes`), the model achieved a recall of approximately 52.59%, identifying more than half of actual subscribers at the selected classification threshold.

### Key Predictors

Based on Permutation Feature Importance and SHAP analysis, the most influential predictors include:

- `nr.employed`
- `poutcome`
- `month`
- `contact`
- `previously_contacted`
- `cons.price.idx`
- `emp.var.rate`
- `euribor3m`
- `campaign`

Permutation Feature Importance identified `nr.employed` as the most influential original input feature, with a mean importance of 0.063.

SHAP analysis further demonstrated the contribution of economic indicators, campaign timing, communication channels, and previous campaign outcomes to the model's predictions.

---

## IV. Conclusion, Business Impact & Recommendations

### Conclusion

- MLP achieved the highest cross-validation PR-AUC of 0.448 and the highest test PR-AUC of 0.461.
- MLP provided the strongest ranking performance, achieving Precision@100 of 0.800 and Lift@100 of 7.102.
- Logistic Regression remained a competitive alternative, achieving a slightly higher ROC-AUC of 0.798.
- Economic indicators, campaign timing, communication channels, and previous campaign outcomes were among the most influential predictive features.
- Permutation Feature Importance and SHAP provided complementary insights into the factors influencing model predictions.

### Recommendations

- **Marketing Department:**
  - Prioritize customers with higher predicted subscription probabilities.
  - Use the top-ranked customer list to allocate limited calling resources.
  - Evaluate campaign performance using Precision@K and Lift@K.

- **Campaign Management:**
  - Consider campaign timing when planning marketing activities.
  - Evaluate communication channels based on their predictive contribution.
  - Use previous campaign outcomes to support customer prioritization.

- **Business Strategy:**
  - Monitor economic indicators such as `nr.employed`, `emp.var.rate`, and `euribor3m`.
  - Incorporate predictive insights into campaign planning and resource allocation.
  - Validate targeting strategies through controlled marketing experiments.

- **Data Science & Analytics:**
  - Retain MLP as the primary predictive model.
  - Maintain Logistic Regression as a competitive baseline.
  - Monitor PR-AUC, Precision@K, Recall@K, and Lift@K to evaluate model performance.

---

## V. Limitations & Next Steps

### Limitations

- The dataset represents historical bank marketing campaigns and may not fully reflect current customer behavior or economic conditions.
- The target variable is imbalanced, with approximately 88.74% negative and 11.26% positive observations.
- The `duration` feature was excluded to prevent target leakage in the pre-call prediction scenario.
- Economic indicators such as `emp.var.rate`, `euribor3m`, and `nr.employed` exhibit strong correlations, which may affect feature importance interpretation.
- Random Forest showed a substantial gap between training and cross-validation performance, suggesting potential overfitting.
- The performance advantage of MLP over Logistic Regression in cross-validation PR-AUC was relatively small.
- Precision@100 and Lift@100 were evaluated using a fixed ranking capacity of 100 customers, which may differ from actual campaign capacity.
- SHAP analysis was performed on 300 test observations and 63 encoded features, so its explanations may not represent every customer profile.
- The evaluation was based on a stratified train-test split rather than a temporal validation strategy.

### Next Steps

- Perform temporal validation to evaluate model generalization on future marketing campaigns.
- Evaluate different outreach capacities using Precision@K, Recall@K, and Lift@K.
- Incorporate campaign costs and expected subscription value into threshold and customer prioritization decisions.
- Investigate additional feature engineering and alternative model configurations.
- Aggregate SHAP values across one-hot encoded categories to improve business-level interpretability.
- Monitor data distribution changes and model performance after deployment.
- Integrate the trained MLP model and preprocessing pipeline into a reusable inference application or marketing dashboard.
- Conduct controlled campaign experiments to measure whether model-assisted targeting improves actual subscription outcomes.

