# HCL-Hackathon

# **Customer Churn Prediction Groupp - 14**

## **1. Overview**

This project performs preprocessing and exploratory analysis on a **Customer Churn dataset**, including:

* Creating an imbalanced sample
* Checking missing values
* Outlier detection
* Plotting distributions (Churn, Age Range, Gender, Subscription Type, Contract Length)
* Encoding categorical features
* Creating new numerical features
* Saving a fully preprocessed dataset
* Generating a correlation heatmap

---

## **2. Tools Used**

| Library      | Purpose                        |
| ------------ | ------------------------------ |
| Pandas       | Data cleaning & manipulation   |
| NumPy        | Numerical operations           |
| Matplotlib   | Bar plots & histograms         |
| Seaborn      | Boxplots, histograms, heatmaps |
| Scikit-learn | Label Encoding                 |

---

## **3. Key Steps**

### **Phase 1: Initial Sampling**

* Loaded raw dataset
* Created a **70% churn – 30% non-churn** sample
* Verified **no missing values**
* Checked outliers using boxplots (none found)
* Saved as `random_data.csv`

---

## **Phase 2: Preprocessing & Visualizations**

### **A. Basic Plots**

1. **Churn Distribution**

   * Bar plot of churn counts

2. **Age Range Distribution**

   * Age grouped into 10-year bins
   * Temporary feature removed after plotting

3. **Gender, Subscription Type, Contract Length**

   * Simple bar charts to inspect category frequencies

4. **Numeric Feature Histograms**

   * Histograms + KDE to check variable distributions

---

### **B. Feature Engineering**

1. **Label Encoding**

   * Encoded: `Gender`, `Subscription Type`

2. **Contract Duration Conversion**

   * Mapped:

     * Annual → 12 months
     * Quarterly → 3 months
     * Monthly → 1 month
   * Dropped original column

3. **Column Cleanup**

   * Removed `CustomerID`
   * Moved `Churn` to last column

4. **Saved final dataset**
   → `Preprocessed_data.csv`

---

### **C. Correlation Heatmap**

* Generated heatmap for the numeric features
* Used to observe relationships affecting churn

---

## **4. Findings**

* No missing values
* No significant outliers
* Balanced visual distributions
* Clean numeric and encoded categorical fields
* Strong correlations visible after preprocessing

---

## **5. Conclusion**

The dataset is now:

* Clean
* Properly encoded
* Fully visualized
* Feature-engineered
* Ready for machine learning models

Output files:

* `random_data.csv`
* `Preprocessed_data.csv`

---
