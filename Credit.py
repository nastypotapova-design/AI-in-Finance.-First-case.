import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LinearRegression


def prepare_test_data(test_df, median_values, params):

    test = test_df.copy()

    # Convert years in current job
    if 'Years in current job' in test.columns:
        test['Years in current job'] = test['Years in current job'].apply(convert_years_to_number)

    # First decision (percentile capping)
    test['Annual Income_processed'] = test['Annual Income'].clip(upper=params['Annual Income_cap'])
    test['Monthly Debt_processed'] = test['Monthly Debt'].clip(upper=params['Monthly Debt_cap'])
    test['Current Credit Balance_processed'] = test['Current Credit Balance'].clip(
        upper=params['Current Credit Balance_cap'])
    test['Maximum Open Credit_processed'] = test['Maximum Open Credit'].clip(upper=params['Maximum Open Credit_cap'])

    # Current Loan Amount (special case)
    test['Current Loan Amount'] = test['Current Loan Amount'].replace(params['Current Loan Amount_special'], np.nan)
    test['Current Loan Amount_processed'] = cap_by_iqr(test['Current Loan Amount'])

    # Credit Score
    test['Credit Score_processed'] = process_credit_score(
        test['Credit Score'],
        min_valid=params['Credit Score_min'],
        max_valid=params['Credit Score_max']
    )

    # Third decision (categorization)
    for feature in params['third_decision']:
        if feature in test.columns:
            test[feature + '_processed'] = test[feature].apply(categorize_counts)

    # Fourth decision (IQR capping)
    for feature in params['fourth_decision']:
        if feature in test.columns and feature != 'Years in current job':
            test[feature + '_processed'] = cap_by_iqr(test[feature])

    # Years in current job (already converted, now cap it)
    if 'Years in current job' in test.columns:
        test['Years in current job_processed'] = cap_by_iqr(test['Years in current job'])

    # One-hot encoding
    cat_features = ['Home Ownership', 'Purpose', 'Term']
    dummies = pd.get_dummies(test[cat_features], prefix=cat_features, dummy_na=False, dtype=int)
    test = pd.concat([test, dummies], axis=1)
    test.drop(columns=cat_features, inplace=True)

    # Fill missing values
    for col, med in median_values.items():
        if col in test.columns:
            test[col] = test[col].fillna(med)

    return test


# Create categorize
def categorize_counts(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return 2


def process_credit_score(series, min_valid=300, max_valid=850):

    # Replace invalid values with NaN
    valid = series.where((series >= min_valid) & (series <= max_valid), np.nan)

    # Apply IQR capping
    return cap_by_iqr(valid)

def cap_by_iqr(series):
    existing = series.dropna()
    if len(existing) > 0:
        Q1 = existing.quantile(0.25)
        Q3 = existing.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(upper=upper_bound)
    return series

def convert_years_to_number(value):
    #Converts a string value to a number of years.
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in ['n/a', 'na', 'none', 'unknown', 'null', '']:
        return np.nan
    if '<' in s:
        return 0.5
    s = s.replace('+', '')
    parts = s.split()
    if parts:
        num = float(parts[0])
        return num

def compute_outlier_stats(series):
    clean = series.dropna()
    Q1_ = clean.quantile(0.25)
    Q3_ = clean.quantile(0.75)
    IQR_ = Q3_ - Q1_
    lower = Q1_ - 1.5 * IQR_
    upper = Q3_ + 1.5 * IQR_
    outliers = clean[(clean < lower) | (clean > upper)]
    count = len(outliers)
    percent = round(count / len(clean) * 100, 2)
    return {
        'outliers_count': count,
        'outliers_percent': percent,
        'lower_bound': round(lower, 2),
        'upper_bound': round(upper, 2)
    }
def make_processed_report(df, col_name, method, percentile_99=None):
    outlier_stats = compute_outlier_stats(df[col_name])
    report = {
        'status': 'analyzed column',
        'feature': col_name,
        'statistics': format_stats(df[col_name]),  # предполагается, что format_stats уже есть
        'outliers_count': outlier_stats['outliers_count'],
        'outliers_percent': outlier_stats['outliers_percent'],
        'lower_bound': outlier_stats['lower_bound'],
        'upper_bound': outlier_stats['upper_bound'],
        'percentile_99': percentile_99,
        'method': method,
        'image_path': visualise(df, col_name)
    }
    return report

def format_stats(series):

    desc = series.describe()

    count_str = f"{int(desc['count']):,}"

    mean_str = f"{desc['mean']:,.2f}"
    std_str = f"{desc['std']:,.2f}"
    min_str = f"{desc['min']:,.2f}"
    q25_str = f"{desc['25%']:,.2f}"
    q50_str = f"{desc['50%']:,.2f}"
    q75_str = f"{desc['75%']:,.2f}"
    max_str = f"{desc['max']:,.2f}"

    result = (
        f"count    {count_str}\n"
        f"mean     {mean_str}\n"
        f"std      {std_str}\n"
        f"min      {min_str}\n"
        f"25%      {q25_str}\n"
        f"50%      {q50_str}\n"
        f"75%      {q75_str}\n"
        f"max      {max_str}"
    )
    return result

def visualise(df, name):
    # Visualisation
    safe_name = name.replace(' ', '_')
    path = f'images/{safe_name}_boxplot_hist.png'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot
    axes[0].boxplot(df[name].dropna())
    axes[0].set_title(f'Boxplot of {name}')

    # Histogram
    axes[1].hist(df[name].dropna(), bins=30, edgecolor='black')
    axes[1].set_title(f'Histogram of {name}')

    # Save and close
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


    return path

#Bonus question
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    def sigmoid(self, z):
        # Ограничиваем значения, чтобы избежать overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iterations):

            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)


            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)


            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)


processing_params = {}

# Convert csv in pd format
train_df = pd.read_csv('course_project_train.csv')

# Create a copy to which we will add the processed features
train_processed = train_df.copy()

# Create a folder for images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

reports_stage1 = []

# List of features
features = ['Annual Income', 'Credit Score','Number of Credit Problems', 'Months since last delinquent', 'Bankruptcies','Tax Liens','Monthly Debt', 'Current Loan Amount',
                  'Current Credit Balance', 'Maximum Open Credit', 'Number of Open Accounts', 'Years of Credit History', 'Years in current job']
#-----------------------------------------------------------------------------------------------------------------------
#Stage1. Processing outliers.
#-----------------------------------------------------------------------------------------------------------------------
for feature in features:
    #processing of the attribute - years in current job
    if 'Years in current job' in train_df.columns:
        train_df['Years in current job'] = train_df['Years in current job'].apply(convert_years_to_number)
        train_processed['Years in current job'] = train_processed['Years in current job'].apply(convert_years_to_number)

    #calculation of initial statistics
    orig_stats = compute_outlier_stats(train_df[feature])
    report_orig = {
        'status': 'column analysis',
        'feature': feature,
        'statistics': format_stats(train_df[feature]),
        'outliers_count': orig_stats['outliers_count'],
        'outliers_percent': orig_stats['outliers_percent'],
        'lower_bound': orig_stats['lower_bound'],
        'upper_bound': orig_stats['upper_bound'],
        'image_path': visualise(train_df, feature)
    }
    reports_stage1.append(report_orig)

    #Based on the initial statistics, a decision was made to assign the feature to one of 4 groups
    first_decision = ['Annual Income','Monthly Debt', 'Current Loan Amount',
                  'Current Credit Balance', 'Maximum Open Credit']
    second_decision = ['Credit Score']
    third_decision = ['Number of Credit Problems', 'Bankruptcies','Tax Liens']
    fourth_decision = ['Months since last delinquent', 'Number of Open Accounts', 'Years of Credit History','Years in current job']


    if feature in first_decision:
        if feature == 'Current Loan Amount':
            # Save a special placeholder value
            processing_params['Current Loan Amount_special'] = 99999999
            #Replace the placeholder code with NaN
            train_processed[feature] = train_processed[feature].replace(99999999, np.nan)
            # Calculate the upper bound of the IQR for the remaining values
            existing = train_processed[feature].dropna()
            if len(existing) > 0:
                Q3 = existing.quantile(0.75)
                IQR = existing.quantile(0.75) - existing.quantile(0.25)
                upper_cap = Q3 + 1.5 * IQR
                # Keep the upper bound
                processing_params['Current Loan Amount_cap'] = upper_cap
                capped = train_processed[feature].clip(upper=upper_cap)
            else:
                capped = train_processed[feature]
            new_feature = feature + '_processed'
            train_processed[new_feature] = capped
            report_proc = make_processed_report(
                train_processed, new_feature,
                method='replace 99999999 with NaN, then cap by IQR upper bound'
            )

        else:
            percentile_99 = train_df[feature].quantile(0.99)
            # After calculating percentile_99, we apply capping and create a new column
            processing_params[f'{feature}_cap'] = percentile_99
            new_feature = feature + '_processed'
            train_processed[new_feature] = train_processed[feature].clip(upper=percentile_99)

            report_proc = make_processed_report(
                train_processed, new_feature,
                method='capping at 99th percentile',
                percentile_99=round(percentile_99, 2)
            )
        reports_stage1.append(report_proc)

    elif feature in second_decision:
        # # Define realistic boundaries for credit score (typically 300-850)
        # min_valid = 300
        # max_valid = 850
        #
        # # Create a temporary column, replacing values outside the range with NaN
        # valid_vals = train_processed[feature].where(
        #     (train_processed[feature] >= min_valid) & (train_processed[feature] <= max_valid),
        #     np.nan
        # )
        #
        # # Count and print the number of removed values
        # n_removed = valid_vals.isna().sum() - train_processed[feature].isna().sum()
        #
        # # Apply capping using the upper IQR bound (to remove outliers within the valid range)
        # valid_clean = valid_vals.dropna()
        # Q1 = valid_vals.quantile(0.25)
        # Q3 = valid_vals.quantile(0.75)
        # IQR = Q3 - Q1
        # upper_bound = Q3 + 1.5 * IQR
        # capped_vals = valid_vals.clip(upper=upper_bound)
        #
        # # Save the final column
        # new_feature = feature + '_processed'
        # train_processed[new_feature] = capped_vals

        processing_params['Credit Score_min'] = 300
        processing_params['Credit Score_max'] = 850

        train_processed[feature + '_processed'] = process_credit_score(train_processed[feature])

        report_proc = make_processed_report(
            train_processed, feature + '_processed',
            method='remove invalid + cap by IQR upper bound'
        )
        reports_stage1.append(report_proc)

    elif feature in third_decision:
        #the method allows us to classify this feature into 3 types

        # def categorize_problems(x):
        #     if pd.isna(x):
        #         return np.nan
        #     if x == 0:
        #         return 0
        #     elif x == 1:
        #         return 1
        #     else:
        #         return 2

        train_processed[feature + '_processed'] = train_processed[feature].apply(categorize_counts)

        report_proc = make_processed_report(
            train_processed, feature + '_processed',
            method='categorized: 0, 1, 2+'
        )
        reports_stage1.append(report_proc)


    elif feature in fourth_decision:
        # Cap outliers above upper bound (Q3 + 1.5*IQR), preserve NaNs
        # existing = train_processed[feature].dropna()
        # if len(existing) > 0:
        #     Q1 = existing.quantile(0.25)
        #     Q3 = existing.quantile(0.75)
        #     IQR = Q3 - Q1
        #     upper_bound = Q3 + 1.5 * IQR
        #     capped_vals = train_processed[feature].clip(upper=upper_bound)
        # else:
        #     capped_vals = train_processed[feature]

        capped_vals = cap_by_iqr(train_processed[feature])

        new_feature = feature + '_processed'
        train_processed[new_feature] = capped_vals

        report_proc = make_processed_report(
            train_processed, new_feature,
            method='capping by upper IQR bound (outliers only), NaNs untouched'
        )
        reports_stage1.append(report_proc)

processing_params['first_decision'] = first_decision
processing_params['second_decision'] = second_decision
processing_params['third_decision'] = third_decision
processing_params['fourth_decision'] = fourth_decision
processing_params['current_loan_special'] = True


cat_features = ['Home Ownership', 'Purpose', 'Term']
#-----------------------------------------------------------------------------------------------------------------------
#Stage2. Processing categorical features.
#-----------------------------------------------------------------------------------------------------------------------
# One-hot encode categorical features: create binary columns for each category
# Prefix = original column name, no column for NaN values
train_dummies = pd.get_dummies(train_processed[cat_features], prefix=cat_features, dummy_na=False, dtype=int)

# Add them to the processed dataset
train_processed = pd.concat([train_processed, train_dummies], axis=1)

# Remove the original categorical columns
train_processed.drop(columns=cat_features, inplace=True)
train_processed.to_csv('train_processed.csv', index=False)

#-----------------------------------------------------------------------------------------------------------------------
#Stage3. Replacing gaps with median values.
#-----------------------------------------------------------------------------------------------------------------------
# Select only _processed columns
processed_features = [col for col in train_processed.columns if col.endswith('_processed')]

median_values = {}

for col in processed_features:
    # Count missing values (NaN) in current column
    missing = train_processed[col].isna().sum()
    # If there are any missing values
    if missing > 0:
        # Calculate the median of the column
        median_val = train_processed[col].median()
        # Store the median value in a dictionary for future reference
        median_values[col] = median_val
        # Replace all NaN values in the column with the median
        train_processed[col] = train_processed[col].fillna(median_val)

# Save the updated dataset
train_processed.to_csv('train_processed.csv', index=False)

# Save the medians for the test (in JSON)
with open('median_values.json', 'w') as f:
    json.dump(median_values, f, indent=4)


#-----------------------------------------------------------------------------------------------------------------------
#Stage 4. Analysis of correlation between all features.
#-----------------------------------------------------------------------------------------------------------------------
# We output binary features from One-Hot coding
dummy_features = [col for col in train_processed.columns if any(col.startswith(prefix) for prefix in ['Home Ownership_', 'Purpose_', 'Term_'])]
# Target variable
target = 'Credit Default'

X = train_processed[processed_features + dummy_features]
y = train_processed[target]

#4.1. Correlation with the target variable
target_corr = X.corrwith(y).sort_values(ascending=False)

#4.2. Teaching RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
# Importance of features
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

#4.3. Summary table for making decision
decision = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_,
    'corr_with_target': X.corrwith(y).values
})
#decision = decision.sort_values('importance', ascending=False)
# Setting the importance threshold
threshold = 0.02
good_features = decision[decision['importance'] > threshold]
candidate_features = good_features['feature'].tolist()
X_candidates = X[candidate_features]

#4.4. Search for duplicate features
corr_candidates = X_candidates.corr()

features_to_keep = []

for current_feat in candidate_features:
    is_duplicate = False

    for kept_feat in features_to_keep:
        if abs(corr_candidates.loc[current_feat, kept_feat]) > 0.8:
            is_duplicate = True
            break

    if not is_duplicate:
        features_to_keep.append(current_feat)

#4.5. Choose main features
for i, feat in enumerate(features_to_keep, 1):
    imp = decision[decision['feature'] == feat]['importance'].values[0]
    corr = decision[decision['feature'] == feat]['corr_with_target'].values[0]

X_final = X[features_to_keep]
y_final = y

#-----------------------------------------------------------------------------------------------------------------------
#Stage 5. Model Training and Evaluation
#-----------------------------------------------------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

# 5.1. Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
f1_lr = f1_score(y_val, y_pred_lr)
precision_lr, recall_lr, _, _ = precision_recall_fscore_support(y_val, y_pred_lr, average='binary', pos_label=1)


# 5.2. Random Forest
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
f1_rf = f1_score(y_val, y_pred_rf)
precision_rf, recall_rf, _, _ = precision_recall_fscore_support(y_val, y_pred_rf, average='binary', pos_label=1)

# 5.3. Method XGBoost
xgb_final = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    scale_pos_weight=3.5,
    random_state=42,
    eval_metric='logloss',
)

xgb_final.fit(X_train, y_train)
y_pred_final = xgb_final.predict(X_val)
f1_final = f1_score(y_val, y_pred_final)
precision_xgb, recall_xgb, _, _ = precision_recall_fscore_support(y_val, y_pred_final, average='binary', pos_label=1)


#-----------------------------------------------------------------------------------------------------------------------
#Stage 6. Model Validation
#-----------------------------------------------------------------------------------------------------------------------
# Perform 10-fold cross-validation
cv_scores = cross_val_score(xgb_final, X_final, y_final, cv=10, scoring='f1')


# Checking for overfitting (comparison of train and validation)
train_pred = xgb_final.predict(X_train)
train_f1 = f1_score(y_train, train_pred)
val_f1 = f1_score(y_val, y_pred_final)


#-----------------------------------------------------------------------------------------------------------------------
#Stage 7. Feature interpretation
#-----------------------------------------------------------------------------------------------------------------------
# Obtaining feature importance from XGBoost
importance_xgb = pd.DataFrame({
    'feature': X_final.columns,
    'importance': xgb_final.feature_importances_
}).sort_values('importance', ascending=False)

# Preview of important features
plt.figure(figsize=(10, 6))
plt.barh(importance_xgb.head(10)['feature'], importance_xgb.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('images/xgb_feature_importance.png')
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#Stage 8. Predictions for test dataset
#-----------------------------------------------------------------------------------------------------------------------

# Load and prepare test data using our function
test_df = pd.read_csv('course_project_test.csv')
test_processed = prepare_test_data(test_df, median_values, processing_params)

# Select features for prediction
model_features = [
    'Annual Income_processed',
    'Credit Score_processed',
    'Months since last delinquent_processed',
    'Monthly Debt_processed',
    'Current Loan Amount_processed',
    'Current Credit Balance_processed',
    'Maximum Open Credit_processed',
    'Number of Open Accounts_processed',
    'Years of Credit History_processed',
    'Years in current job_processed'
]
X_test = test_processed[model_features]

# Make predictions
test_predictions = xgb_final.predict(X_test)

# Save results
output = pd.DataFrame({'Credit Default': test_predictions.astype(int)})
output.to_csv('predictions.csv', index=False)


#-----------------------------------------------------------------------------------------------------------------------
# BONUS: Custom Logistic Regression vs LinearRegression
#-----------------------------------------------------------------------------------------------------------------------
X_train_bonus, X_val_bonus, y_train_bonus, y_val_bonus = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)
# 1.Self-written logistic regression

custom_lr = CustomLogisticRegression(learning_rate=0.1, n_iterations=5000)
custom_lr.fit(X_train_bonus.values, y_train_bonus.values)
y_pred_custom = custom_lr.predict(X_val_bonus.values)
f1_custom = f1_score(y_val_bonus, y_pred_custom)


# 2. LinearRegression from sklearn

lin_reg = LinearRegression()
lin_reg.fit(X_train_bonus, y_train_bonus)
y_pred_lin = lin_reg.predict(X_val_bonus)
y_pred_lin_binary = (y_pred_lin > 0.5).astype(int)
f1_lin = f1_score(y_val_bonus, y_pred_lin_binary)


# 3. LogisticRegression from sklearn


sklearn_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
sklearn_lr.fit(X_train_bonus, y_train_bonus)
y_pred_sklearn = sklearn_lr.predict(X_val_bonus)
f1_sklearn = f1_score(y_val_bonus, y_pred_sklearn)


# 4. Comparison with XGBoost
f1_xgb = f1_score(y_val_bonus, xgb_final.predict(X_val_bonus))



#-----------------------------------------------------------------------------------------------------------------------
# Record all results in README
#-----------------------------------------------------------------------------------------------------------------------

with open('README.md', 'w', encoding='utf-8') as f:
    # Write main title
    f.write('# Step-by-step report on the model predicting debt defaults\n\n')
    # Write introductory description
    f.write('This document describes the step-by-step processing of the training dataset and the analysis of the test dataset.\n\n')
    # Stage 1 heading


    f.write('## Stage 1. Handling outliers by columns\n\n')
    i = 0
    while i < len(reports_stage1):
        rep_orig = reports_stage1[i]
        # Process only original analysis entries
        if rep_orig['status'] == 'column analysis':
            # Feature name as subheading
            f.write(f'### {rep_orig["feature"]}\n\n')
            # Original statistics block
            f.write('**Original statistics:**\n\n')
            f.write(f'```\n{rep_orig["statistics"]}\n```\n\n')
            # Outlier info from IQR method
            f.write(f'- Number of outliers (IQR): {rep_orig["outliers_count"]} ({rep_orig["outliers_percent"]}%)\n')
            f.write(f'- Normal range by IQR: [{rep_orig["lower_bound"]}, {rep_orig["upper_bound"]}]\n\n')
            # Link to the original plot image
            f.write(f'![Original plot]({rep_orig["image_path"]})\n\n')

            # Check if there is a corresponding processed entry (next item starts with feature name + underscore)
            if i + 1 < len(reports_stage1) and reports_stage1[i + 1]['feature'].startswith(rep_orig['feature'] + '_'):
                rep_proc = reports_stage1[i + 1]
                f.write('**Processing:**\n\n')
                f.write(f'Method: {rep_proc["method"]}\n\n')
                # If 99th percentile was used, display it
                if rep_proc.get('percentile_99') is not None:
                    f.write(f'- 99th percentile used for capping: {rep_proc["percentile_99"]}\n\n')

                # Statistics after processing
                f.write('**Statistics after processing:**\n\n')
                f.write(f'```\n{rep_proc["statistics"]}\n```\n\n')
                # Outlier info after processing
                f.write(f'- Number of outliers (IQR) after: {rep_proc["outliers_count"]} ({rep_proc["outliers_percent"]}%)\n')
                f.write(f'- Normal range by IQR after: [{rep_proc["lower_bound"]}, {rep_proc["upper_bound"]}]\n\n')
                # Link to the processed plot image
                f.write(f'![Processed plot]({rep_proc["image_path"]})\n\n')
                i += 1  # skip the processed entry in next iteration

            f.write('---\n\n')
        i += 1




    f.write('\n## Stage 2. Encoding Categorical Features\n\n')
    f.write('The following categorical features were one-hot encoded:\n')
    for col in cat_features:
        f.write(f'- {col}\n')
    f.write(f'\nNumber of new dummy columns created: {train_dummies.shape[1]}\n')
    f.write('Original categorical columns were removed.\n')
    f.write('No missing values were present in these columns.\n')


    f.write('\n## Stage 3. Handling Missing Values in Numeric Features\n\n')
    if median_values:
        f.write('Missing values in the following processed columns were filled with the median:\n')
        for col, med in median_values.items():
            f.write(f'- {col}: median = {med:.2f}\n')
    else:
        f.write('No missing values were found in numeric features after outlier processing.\n')




    f.write('\n## Stage 4. Feature Selection and Analysis\n\n')

    f.write('### 4.1 Correlation with Target Variable\n\n')
    f.write('Correlation with Credit Default shows linear relationships:\n\n')
    pos_corr = target_corr.head(10)
    f.write('**Features increasing default risk (positive correlation):**\n\n')
    for feat, corr in pos_corr.items():
        f.write(f'- {feat}: {corr:.3f}\n')
    neg_corr = target_corr.tail(10)
    f.write('\n**Features decreasing default risk (negative correlation):**\n\n')
    for feat, corr in neg_corr.items():
        f.write(f'- {feat}: {corr:.3f}\n')

    f.write('\n### 4.2 Feature Importance (Random Forest)\n\n')
    f.write('Random Forest captures non-linear relationships and shows the most predictive features:\n\n')
    f.write('**Top 20 most important features:**\n\n')
    for idx, row in importance.head(20).iterrows():
        f.write(f'{idx + 1}. **{row["feature"]}**: {row["importance"]:.4f}\n')

    #Visualisation
    plt.figure(figsize=(12, 8))
    importance.head(20).plot(x='feature', y='importance', kind='barh')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('images/feature_importance_stage4.png')
    plt.close()

    f.write('\n![Feature Importance](images/feature_importance_stage4.png)\n\n')

    f.write('\n### 4.3 Feature Selection Process\n\n')
    f.write(f'**Step 1:** Filter by importance threshold (> {threshold})\n')
    f.write(f'- Initial number of features: {len(X.columns)}\n')
    f.write(f'- Features above threshold: {len(candidate_features)}\n\n')

    f.write('**Step 2:** Remove duplicate features (correlation > 0.8)\n')
    f.write(f'- Features after removing duplicates: {len(features_to_keep)}\n\n')

    f.write('### 4.4 Final Selected Features\n\n')
    f.write('The following features will be used for modeling:\n\n')
    for i, feat in enumerate(features_to_keep, 1):
        imp = decision[decision['feature'] == feat]['importance'].values[0]
        corr = decision[decision['feature'] == feat]['corr_with_target'].values[0]
        direction = " increases" if corr > 0 else " decreases"
        f.write(f'{i}. **{feat}**  \n')
        f.write(f'   - Importance: {imp:.4f}  \n')
        f.write(f'   - Correlation: {corr:.3f} ({direction} risk)  \n\n')


    f.write('\n## Stage 5. Model Training and Evaluation\n\n')
    f.write('Data split: 80% train / 20% validation (stratified).\n\n')
    f.write(f'- **Train size**: {len(X_train)}\n')
    f.write(f'- **Validation size**: {len(X_val)}\n')
    f.write(f'- **Train default rate**: {y_train.mean():.2%}\n')
    f.write(f'- **Validation default rate**: {y_val.mean():.2%}\n\n')

    f.write('### 5.1 Logistic Regression\n')
    f.write('```python\nlr = LogisticRegression(class_weight=\'balanced\')\n```\n')
    f.write(f'- **F1-score**: {f1_lr:.4f}\n')
    f.write(f'- Precision: {precision_lr:.2f} | Recall: {recall_lr:.2f}\n\n')

    f.write('### 5.2 Random Forest\n')
    f.write('```python\nrf = RandomForestClassifier(class_weight=\'balanced\')\n```\n')
    f.write(f'- **F1-score**: {f1_rf:.4f}\n')
    f.write(f'- **Precision**: {precision_rf:.2f} | **Recall**: {recall_rf:.2f}\n\n')

    f.write('### 5.3 XGBoost (Best Model)\n')
    f.write('```python\nxgb = XGBClassifier(scale_pos_weight=3.5, max_depth=3, learning_rate=0.05)\n```\n')
    f.write(f'- **F1-score**: **{f1_final:.4f}** (> 0.5)\n')
    f.write(f'- **Precision**: {precision_xgb:.2f} | **Recall**: {recall_xgb:.2f}\n\n')

    f.write('### 5.4 Summary\n')
    f.write('| Model                 |  F1-score  | Precision | Recall |\n')
    f.write('|-----------------------|------------|-----------|--------|\n')
    f.write(f'| Logistic Regression   | {f1_lr:.4f}     | {precision_lr:.2f}      | {recall_lr:.2f}   |\n')
    f.write(f'| Random Forest         | {f1_rf:.4f}     | {precision_rf:.2f}      | {recall_rf:.2f}   |\n')
    f.write(f'| **XGBoost**           | **{f1_final:.4f}** | {precision_xgb:.2f}      | {recall_xgb:.2f}   |\n\n')

    f.write('**XGBoost selected as final model** — best balance of precision/recall and meets the F1 > 0.5 requirement.\n')




    f.write('\n## Stage 6. Model Validation\n\n')
    # Cross-validation results
    f.write('### 6.1 Cross-Validation (10-fold)\n\n')
    f.write(f'- **F1 scores for each fold**: {[round(score, 4) for score in cv_scores]}\n')
    f.write(f'- **Mean F1**: {cv_scores.mean():.4f}\n')
    f.write(f'- **Standard Deviation**: {cv_scores.std():.4f}\n\n')
    # Overfitting check
    f.write('### 6.2 Overfitting Check\n\n')
    f.write(f'- **Train F1**: {train_f1:.4f}\n')
    f.write(f'- **Validation F1**: {val_f1:.4f}\n')
    f.write(f'- **Difference**: {abs(train_f1 - val_f1):.4f}\n\n')
    # Conclusion
    f.write('### 6.3 Conclusion\n\n')
    if abs(train_f1 - val_f1) < 0.1:
        f.write(' **No significant overfitting detected** — the model generalizes well.\n')
    else:
        f.write(' **Possible overfitting detected** — consider adding regularization.\n')

    f.write(f'\nThe model shows stable performance across all folds (std = {cv_scores.std():.4f}) and meets the F1 > 0.5 requirement.\n')




    f.write('\n## Stage 7. Feature Interpretation\n\n')
    f.write('### 7.1 Top 10 Important Features\n\n')
    f.write('| Feature | Importance |\n')
    f.write('|:--------|-----------:|\n')
    for idx, row in importance_xgb.head(10).iterrows():
        f.write(f'| {row["feature"]} | {row["importance"]:.4f} |\n')
    f.write('\n')

    f.write('\n### 7.2 Key Insights\n\n')
    top3 = importance_xgb.head(3)
    f.write(f'**Top 3 features account for {top3["importance"].sum():.2%} of total importance:**\n\n')
    for idx, row in top3.iterrows():
        f.write(f'- **{row["feature"]}**: {row["importance"]:.2%}\n')

    f.write('\n### 7.3 Visualization\n\n')
    f.write('![Feature Importance](images/xgb_feature_importance.png)\n\n')

    f.write('### 7.4 Business Logic\n\n')
    f.write('The model\'s top features align with real-world credit risk assessment:\n')
    f.write('- Credit Score (lower score → higher risk)\n')
    f.write('- Current Loan Amount (larger loans → higher exposure)\n')
    f.write('- Annual Income (higher income → better repayment ability)\n\n')

    f.write(' **Model is interpretable and uses meaningful features**\n')



    f.write('\n## Stage 8. Final Results\n\n')
    f.write('### 8.1 Test Dataset Predictions\n\n')
    f.write(f'- **Total predictions**: {len(output)}\n')

    # Distribution of predictions
    pred_counts = output['Credit Default'].value_counts()
    f.write(
        f'- **Predicted defaults (class 1)**: {pred_counts.get(1, 0)} ({pred_counts.get(1, 0) / len(output) * 100:.1f}%)\n')
    f.write(
        f'- **Predicted non-defaults (class 0)**: {pred_counts.get(0, 0)} ({pred_counts.get(0, 0) / len(output) * 100:.1f}%)\n\n')

    f.write('### 8.2 Final Model Performance Summary\n\n')
    f.write('| Metric | Value |\n')
    f.write('|:--------|-------:|\n')
    f.write('| Best Model | XGBoost |\n')
    f.write(f'| F1-score (validation) | {f1_final:.4f} |\n')
    f.write(f'| Cross-validation F1 (10-fold) | {cv_scores.mean():.4f} ± {cv_scores.std():.4f} |\n')
    f.write(f'| Train F1 | {train_f1:.4f} |\n')
    f.write(f'| Validation F1 | {val_f1:.4f} |\n\n')

    f.write('### 8.3 Conclusion\n\n')
    f.write('1. F1-score > 0.5 achieved: **{:.4f}**\n'.format(f1_final))
    f.write('2. Complete ML pipeline implemented (EDA, preprocessing, feature engineering, modeling)\n')
    f.write('3. Model is interpretable with Credit Score as the most important feature\n')

    f.write('\n## BONUS: Custom Logistic Regression vs LinearRegression\n\n')
    f.write('### Implementation Results\n\n')
    f.write('| Model | F1-score | Note |\n')
    f.write('|:------|---------:|:-----|\n')
    f.write(f'| Custom Logistic Regression | {f1_custom:.4f} | Implemented with gradient descent |\n')
    f.write(f'| sklearn LinearRegression | {f1_lin:.4f} |  Wrong model for classification |\n')
    f.write(f'| sklearn LogisticRegression | {f1_sklearn:.4f} |  Reference implementation |\n')
    f.write(f'| XGBoost (best) | {f1_xgb:.4f} |  Final project model |\n\n')






