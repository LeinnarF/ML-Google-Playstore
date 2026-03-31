# Machine Learning Notebook Evaluation Report
## Google Play Store App Analysis

**Date:** March 31, 2026  
**Notebook:** `machine_learning.ipynb`  
**Evaluator:** GitHub Copilot CLI

---

## Executive Summary

This notebook demonstrates a comprehensive machine learning approach to analyzing Google Play Store app data through two main objectives: (1) predicting app success using Random Forest classification and (2) dimensionality reduction using Principal Component Analysis (PCA) with KNN clustering. While the notebook shows solid analytical foundations and produces interpretable results, there are several areas for improvement in terms of code organization, model evaluation depth, and best practices.

**Overall Assessment:** ⭐⭐⭐⭐ (4/5 stars)

---

## 1. Structure & Organization

### Strengths
- **Clear objectives**: The notebook is well-organized into distinct sections with clear problem statements
- **Progressive complexity**: Starts with data preprocessing, moves to classification, then advanced dimensionality reduction
- **Interactive elements**: Includes a user input function for predictions (commented out)
- **Markdown documentation**: Good use of markdown cells to explain objectives and subproblems

### Areas for Improvement
- **Code modularity**: Many code cells are long and perform multiple operations. Consider breaking them into smaller, focused cells
- **Function definitions**: Repetitive operations (like plotting) should be extracted into reusable functions
- **Import organization**: All imports are in the first cell, but some libraries (like `warnings`) are imported later
- **Section headers**: Would benefit from more descriptive headers and intermediate summaries

**Recommendation:** Create a table of contents cell and add more intermediate summary cells explaining findings

---

## 2. Data Preprocessing

### Strengths
- **Smart feature engineering**: Created `Size_varies` binary feature to capture "Varies with device" information
- **Category-based imputation**: Uses grouped median for missing values (sophisticated approach)
- **Fallback strategy**: Has a global median fallback for edge cases
- **Target variable**: Clear definition of success (>100k installs)

### Areas for Improvement
```python
# Current approach
df['Size'].fillna(df['Size'].median(), inplace=True)
```

**Issues:**
1. **In-place modification**: Modifies original dataframe; should create a copy for ML processing
2. **No data validation**: Missing checks for other potential data quality issues
3. **No feature scaling**: Size and Price are on different scales but no normalization applied for Random Forest
4. **Missing value documentation**: No reporting of how many values were imputed

**Recommendations:**
```python
# Create a preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Document preprocessing steps
preprocessing_report = {
    'missing_size_values': df['Size'].isna().sum(),
    'varies_with_device_count': df['Size_varies'].sum(),
    'size_imputation_method': 'category_median'
}
print(f"Preprocessing Report: {preprocessing_report}")
```

---

## 3. Model 1: Random Forest Classifier (Hit/Miss Prediction)

### Strengths
- **Appropriate model choice**: Random Forest is excellent for this tabular classification task
- **Feature importance analysis**: Examines which features matter most
- **Good feature selection**: Uses relevant categorical and numerical features
- **Train-test split**: Properly separates data (80/20 split with fixed random_state)

### Current Performance
```
Accuracy: 70%
Precision (Hit): 64%
Recall (Hit): 60%
F1-Score (Hit): 62%
```

### Critical Issues

#### Issue 1: No Cross-Validation
**Problem:** Single train-test split doesn't validate model stability
```python
# Current approach
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=67)
```

**Solution:**
```python
from sklearn.model_selection import cross_val_score, cross_validate

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_classifier, X_encoded, y, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

#### Issue 2: No Hyperparameter Tuning
**Problem:** Uses default RandomForest parameters (n_estimators=100)
```python
# Current approach
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=67, n_jobs=-1)
```

**Solution:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=67, n_jobs=-1),
    param_distributions,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=67
)
random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV F1 score: {random_search.best_score_:.3f}")
```

#### Issue 3: Class Imbalance Not Addressed
**Problem:** Model performance differs significantly between classes (Miss: 75% vs Hit: 62% F1)

**Solution:**
```python
# Check class distribution
print(f"Class distribution:\n{y.value_counts(normalize=True)}")

# Address imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
rf_classifier = RandomForestClassifier(
    class_weight='balanced',  # or dict(enumerate(class_weights))
    random_state=67,
    n_jobs=-1
)
```

#### Issue 4: Limited Model Evaluation Metrics
**Problem:** Only shows classification report, no ROC curve, confusion matrix visualization, or feature importance plot

**Solution:**
```python
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ROC Curve
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - App Success Prediction')
plt.legend()
plt.show()

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_classifier, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()
```

### Feature Importance Insights
- **Size dominates** (58.8%): App size is the strongest predictor
- **Type_Paid** (3.7%): Whether an app is paid matters
- **Price** (3.3%): Direct relationship with success
- **Category_GAME** (2.5%): Game category is significant

**Recommendation:** Consider non-linear transformations or interactions between Size and Category

---

## 4. Interactive Prediction Function

### Strengths
- **User-friendly**: Good idea for practical application
- **Input validation**: Handles categorical encoding properly
- **Probability display**: Shows confidence percentage

### Areas for Improvement

**Issue 1: Silent Failures**
```python
# Current approach - silently handles unknown categories
if category_col in training_columns:
    input_data[category_col] = 1
```

**Solution:**
```python
# Warn user about unknown categories
known_categories = [col.replace('Category_', '') for col in training_columns if col.startswith('Category_')]
if category not in known_categories:
    print(f"⚠️ Warning: '{category}' not in training data. Available: {known_categories}")
    return
```

**Issue 2: Hardcoded Column Name**
```python
# This line has incorrect column name
if varies == 'y' and 'Size_Varies with device' in training_columns:
    input_data['Size_Varies with device'] = 1
```

**Should be:** `'Size_varies'` (snake_case, as defined in preprocessing)

**Issue 3: Missing Size Input**
The function doesn't ask for app size, which is the most important feature (58.8% importance)!

**Solution:**
```python
size = float(input("Enter App Size in KB (or 0 if varies): "))
if size == 0:
    input_data['Size_varies'] = 1
    # Use median size for the category
    input_data['Size'] = df.groupby('Category')['Size'].median().get(category, df['Size'].median())
else:
    input_data['Size'] = size
```

---

## 5. Model 2: Principal Component Analysis (PCA)

### Strengths
- **Feature engineering**: Smart log transformations for skewed distributions
- **Good interpretability**: Named PCs meaningfully ("Viral Reach", "Premium Penalty")
- **Comprehensive features**: Uses 5 continuous features
- **Proper scaling**: StandardScaler applied before PCA
- **Explained variance**: 2 components capture 95.2% of variance (excellent)

### Current Approach
```python
continuous_features = [
  'Log_Installs',
  'Rating',
  'Log_Reviews',
  'Price',
  'Log_Size'
]
```

### Areas for Improvement

#### Issue 1: No Scree Plot or Explained Variance Visualization
**Problem:** Doesn't show why 2 components were chosen

**Solution:**
```python
# Create scree plot
pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='k', linestyle='--', label='95% threshold')
plt.title('Cumulative Variance Explained')
plt.legend()
plt.tight_layout()
plt.show()
```

#### Issue 2: Component Interpretation Not Fully Explained
**Problem:** PC1 and PC2 are named but component loadings aren't shown

**Solution:**
```python
# Show component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=continuous_features
)
print("\nPCA Component Loadings:")
print(loadings.round(3))

# Visualize loadings
loadings.plot(kind='bar', figsize=(10, 6))
plt.title('PCA Component Loadings')
plt.ylabel('Loading Value')
plt.xlabel('Feature')
plt.legend(['PC1: Viral Reach', 'PC2: Premium Penalty'])
plt.tight_layout()
plt.show()
```

#### Issue 3: Outlier Analysis Could Be More Robust
**Problem:** Simple threshold (PC2 > 10) for outliers

**Solution:**
```python
from scipy import stats

# Use statistical methods for outlier detection
z_scores = np.abs(stats.zscore(pca_df[['PC1 (Viral Reach)', 'PC2 (Premium Penalty)']]))
outliers = (z_scores > 3).any(axis=1)

print(f"Number of statistical outliers: {outliers.sum()}")
pca_df['is_outlier'] = outliers
```

---

## 6. Model 3: KNN Classifier on PCA Components

### Strengths
- **Creative visualization**: Excellent decision boundary and probability heatmap plots
- **Proper data handling**: Correctly slices 2D components
- **Parameter documentation**: K=5 is explicitly stated
- **Visual quality**: High-resolution meshgrid for smooth heatmaps

### Critical Issues

#### Issue 1: Data Leakage!
**MAJOR PROBLEM:**
```python
# Line 1: Create features and target
X_knn = X_pca_2d[:, :2]  
y_knn = (df_pca['Installs'] >= 100000).astype(int)

# Line 2: Fit on ALL data
knn.fit(X_knn, y_knn)

# Lines 3-4: THEN split and report
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=67)
knn.fit(X_train_knn, y_train_knn)
print(classification_report(y_test_knn, knn.predict(X_test_knn)))
```

The visualizations (decision boundary and heatmap) are created using a model trained on ALL data, then the model is retrained for evaluation. This is confusing and the visualizations are misleading.

**Solution:**
```python
# 1. Split data FIRST
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=67
)

# 2. Fit on training data ONLY
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_knn, y_train_knn)

# 3. Evaluate on test data
print("Test Set Performance:")
print(classification_report(y_test_knn, knn.predict(X_test_knn)))

# 4. For visualization, create meshgrid based on training data range
x_min, x_max = X_train_knn[:, 0].min() - 1, X_train_knn[:, 0].max() + 1
y_min, y_max = X_train_knn[:, 1].min() - 1, X_train_knn[:, 1].max() + 1
```

#### Issue 2: No K-Value Optimization
**Problem:** K=5 is arbitrary; no justification

**Solution:**
```python
from sklearn.model_selection import cross_val_score

# Test different K values
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_knn, y_train_knn, cv=5, scoring='f1')
    cv_scores.append(scores.mean())

# Find optimal K
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")

# Plot K vs F1 score
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, 'bo-')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Cross-validated F1 Score')
plt.title('KNN: Optimal K Selection')
plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
plt.legend()
plt.show()
```

#### Issue 3: KNN on PCA vs Original Random Forest Not Compared
**Problem:** No comparison between the three approaches:
1. Random Forest on original features (70% accuracy)
2. Random Forest on PCA components (not tested)
3. KNN on PCA components (tested but not reported in the final cell)

**Solution:**
```python
# Create comparison dataframe
results = {
    'Model': ['Random Forest (Original)', 'Random Forest (PCA)', 'KNN (PCA)'],
    'Accuracy': [0.70, rf_pca_acc, knn_pca_acc],
    'F1-Score': [0.70, rf_pca_f1, knn_pca_f1],
    'Training Time': [rf_time, rf_pca_time, knn_time]
}
results_df = pd.DataFrame(results)
print(results_df)
```

---

## 7. Code Quality & Best Practices

### Issues Found

#### Issue 1: Magic Numbers
```python
# What does 100_000 represent?
df['Success'] = (df['Installs'] > 100_000).astype(int)

# Better:
SUCCESS_THRESHOLD_INSTALLS = 100_000  # Define as constant
df['Success'] = (df['Installs'] > SUCCESS_THRESHOLD_INSTALLS).astype(int)
```

#### Issue 2: Inconsistent Naming
- `df_pca` vs `df` (both modify the same data structure conceptually)
- `X_knn` vs `X_encoded` (inconsistent suffixes)
- `Size_varies` vs `Log_Installs` (inconsistent conventions)

**Recommendation:** Adopt consistent naming:
- Use `df_raw` for original data
- Use `df_processed` for feature-engineered data
- Use `X_train_rf`, `X_train_knn` to distinguish datasets

#### Issue 3: No Documentation Strings
Functions lack docstrings explaining parameters and return values

```python
def predict_app_success(model, training_columns):
    """
    Predict app success based on user input features.
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    training_columns : pd.Index
        Column names from training data for proper encoding
        
    Returns:
    --------
    None (prints prediction and confidence to console)
    """
    # ... function code ...
```

#### Issue 4: Commented-Out Code
```python
# Run Script
# while True:
#     predict_app_success(rf_classifier, X_encoded.columns)
#     ...
```

**Recommendation:** Either remove or replace with proper execution cell:
```python
# Uncomment to run interactive prediction:
# predict_app_success(rf_classifier, X_encoded.columns)
```

#### Issue 5: Warnings Suppression Without Explanation
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

**Better approach:**
```python
# Suppress specific warnings from matplotlib during user input
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
```

---

## 8. Visualization Quality

### Strengths
- **Professional aesthetics**: Good use of color, sizing, and labels
- **Multiple visualization types**: Scatter, heatmap, decision boundaries
- **Clear legends**: Well-labeled axes and titles
- **High resolution**: Appropriate figure sizes and DPI

### Areas for Improvement

#### Issue 1: Missing Key Plots
1. **Feature correlation heatmap** - would show multicollinearity
2. **Feature importance bar chart** - currently only shows DataFrame
3. **Actual vs Predicted scatter** - for regression-style evaluation
4. **Learning curves** - to diagnose overfitting

#### Issue 2: Color Accessibility
Current palette may not be colorblind-friendly:
```python
cmap_points = {0: '#D00000', 1: '#00B4D8'}  # Red-Blue
```

**Recommendation:**
```python
# Use colorblind-friendly palette
cmap_points = {0: '#E69F00', 1: '#56B4E9'}  # Orange-Blue (colorblind-safe)
```

#### Issue 3: No Saved Figures
All plots are shown but not saved

**Solution:**
```python
plt.savefig('figures/pca_visualization.png', dpi=300, bbox_inches='tight')
```

---

## 9. Missing Components

### Critical Missing Elements

1. **Model Persistence**
```python
import joblib

# Save trained models
joblib.dump(rf_classifier, 'models/random_forest_classifier.pkl')
joblib.dump(pca, 'models/pca_transformer.pkl')
joblib.dump(scaler, 'models/standard_scaler.pkl')
```

2. **Reproducibility Section**
```python
# Set all random seeds for reproducibility
np.random.seed(67)
import random
random.seed(67)
import os
os.environ['PYTHONHASHSEED'] = '67'
```

3. **Data Quality Report**
```python
# Missing values summary
missing_data = df.isnull().sum()
print("Missing Data Summary:")
print(missing_data[missing_data > 0])

# Duplicate check
print(f"\nDuplicate rows: {df.duplicated().sum()}")
```

4. **Model Comparison Summary**
Should include a final section comparing all approaches

5. **Business Recommendations**
No actionable insights for app developers based on findings

6. **Error Analysis**
No analysis of misclassified examples

---

## 10. Recommendations Summary

### High Priority (Must Fix)

1. **Fix data leakage in KNN visualization** - Train on training set only
2. **Add cross-validation** - For model reliability assessment
3. **Implement hyperparameter tuning** - To optimize performance
4. **Fix interactive prediction function** - Include Size input, fix column names
5. **Add model comparison section** - Compare all three approaches

### Medium Priority (Should Fix)

6. **Address class imbalance** - Use class weights or SMOTE
7. **Optimize KNN K-value** - Use cross-validation
8. **Add component loading analysis** - For PCA interpretation
9. **Create model persistence** - Save trained models
10. **Add confusion matrix visualizations** - Better error understanding

### Low Priority (Nice to Have)

11. **Refactor into functions** - Improve code reusability
12. **Add docstrings** - Better code documentation
13. **Create figure directory** - Save all plots
14. **Add business insights section** - Actionable recommendations
15. **Improve naming consistency** - Follow single convention

---

## 11. Strengths to Maintain

1. ✅ **Clear problem definition** with well-structured objectives
2. ✅ **Good feature engineering** (log transforms, binary indicators)
3. ✅ **Excellent visualizations** (heatmaps, decision boundaries)
4. ✅ **PCA dimensionality reduction** successfully retains 95% variance
5. ✅ **Feature importance analysis** provides interpretable insights
6. ✅ **Interactive prediction function** (good idea, needs refinement)

---

## 12. Final Scoring

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| **Code Quality** | 3/5 | 5 | Good structure, but lacks modularity and documentation |
| **Model Performance** | 3/5 | 5 | Decent baseline (70%), but no optimization or comparison |
| **Methodology** | 3.5/5 | 5 | Sound approach with data leakage issue in KNN |
| **Visualization** | 4.5/5 | 5 | Excellent plots, missing some key diagnostic charts |
| **Documentation** | 4/5 | 5 | Clear objectives, but lacks technical comments |
| **Reproducibility** | 2/5 | 5 | Fixed random_state, but no full seed setting or saved models |
| **Innovation** | 4/5 | 5 | Creative use of PCA + KNN visualization |
| **Practical Value** | 3/5 | 5 | Interactive function good, but needs business insights |

**Overall: 27/40 (67.5%) - Solid B Grade**

---

## 13. Suggested Next Steps

### Immediate Actions (Next Session)
1. Create a new cell fixing the KNN data leakage issue
2. Add cross-validation to Random Forest
3. Fix the `predict_app_success` function bugs
4. Add confusion matrix visualization

### Short-term Improvements (This Week)
5. Implement hyperparameter tuning with RandomizedSearchCV
6. Add model comparison section
7. Create PCA component loading visualization
8. Optimize KNN K-value

### Long-term Enhancements (Future Work)
9. Try additional models (XGBoost, LightGBM, Neural Networks)
10. Feature engineering: create interaction terms
11. Time-series analysis if data has temporal component
12. Deploy model as web app using Streamlit/Flask

---

## 14. Code Snippets for Quick Wins

### Add This Cell After Random Forest Training
```python
# Cross-validation evaluation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_classifier, X_encoded, y, cv=5, scoring='f1_weighted')
print(f"5-Fold CV F1 Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Confusion Matrix Visualization
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(rf_classifier, X_test, y_test, 
                                      display_labels=['Miss', 'Hit'])
plt.title('Random Forest Confusion Matrix')
plt.show()
```

### Add This Cell for PCA Component Analysis
```python
# PCA Component Loadings
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1: Viral Reach', 'PC2: Premium Penalty'],
    index=continuous_features
)

print("Component Loadings:")
print(loadings_df.round(3))

# Visualize
loadings_df.plot(kind='barh', figsize=(10, 6))
plt.title('PCA Component Loadings')
plt.xlabel('Loading Value')
plt.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### Fixed KNN Training Cell
```python
# FIXED VERSION - No data leakage

# 1. Split FIRST
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=67
)

# 2. Fit on training data only
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_knn, y_train_knn)

# 3. Evaluate
print("KNN Classification Report (Test Set):")
print(classification_report(y_test_knn, knn.predict(X_test_knn), 
                          target_names=['Flop', 'Hit']))

# 4. Create visualization from training data
h = 0.05
x_min, x_max = X_train_knn[:, 0].min() - 1, X_train_knn[:, 0].max() + 1
y_min, y_max = X_train_knn[:, 1].min() - 1, X_train_knn[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ... rest of visualization code ...
```

---

## Conclusion

This notebook demonstrates **strong analytical skills** and **creative problem-solving** through the combination of classification, dimensionality reduction, and visualization techniques. The major areas needing improvement are:

1. **Model validation rigor** (cross-validation, hyperparameter tuning)
2. **Data leakage prevention** (fix KNN training workflow)
3. **Code organization** (functions, constants, documentation)
4. **Comparative analysis** (benchmark multiple approaches)

With these improvements, this could easily become a **portfolio-worthy project** demonstrating professional-level ML engineering skills.

**Estimated time to implement high-priority fixes: 2-3 hours**

---

## References & Resources

For implementing the suggested improvements, refer to:

- **Scikit-learn Documentation**: https://scikit-learn.org/stable/
- **Cross-validation**: https://scikit-learn.org/stable/modules/cross_validation.html
- **Hyperparameter Tuning**: https://scikit-learn.org/stable/modules/grid_search.html
- **PCA Tutorial**: https://scikit-learn.org/stable/modules/decomposition.html#pca
- **Imbalanced Learning**: https://imbalanced-learn.org/stable/
- **Model Persistence**: https://scikit-learn.org/stable/model_persistence.html

---

**Report Generated:** March 31, 2026  
**Notebook Version:** Current (1690 lines)  
**Next Review:** After implementing high-priority fixes
