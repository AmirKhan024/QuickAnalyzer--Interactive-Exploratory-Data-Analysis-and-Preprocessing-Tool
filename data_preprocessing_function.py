import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats


def remove_selected_columns(df,columns_remove):
    return df.drop(columns=columns_remove)

# Create a function to remove rows with missing values in specific columns
def remove_rows_with_missing_data(df, columns):
    if columns:
        df = df.dropna(subset=columns)
        return df

# Create a function to fill missing data with mean, median, or mode (for numerical columns)
def fill_missing_data(df, columns, method):
    for column in columns:
        if method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif method == 'mode':
            mode_val = df[column].mode().iloc[0]
            df[column].fillna(mode_val, inplace=True)
    return df

# Function to detect duplicates
# Function to detect duplicates and return duplicate rows
def detect_duplicates(df):
    duplicate_flags = df.duplicated()  # This returns True for all but the first duplicate
    duplicate_rows = df[duplicate_flags]
    duplicate_count = duplicate_flags.sum()
    return duplicate_count, duplicate_rows



# Function to remove duplicates
# Function to remove duplicates and reset count
def remove_duplicates(df):
    duplicate_count, duplicate_rows = detect_duplicates(df)  # Get duplicate rows before deleting
    if duplicate_count > 0:
        df = df.drop_duplicates()  # Remove duplicates
        st.success(f"✅ {duplicate_count} duplicate rows deleted.")
        duplicate_count = 0  # Reset count to 0 after deletion
    else:
        st.info("No duplicate rows found.")
    
    return df, duplicate_count


def one_hot_encode(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)
    return df


def label_encode(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df



def standard_scale(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def min_max_scale(df, columns, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    df[columns] = scaler.fit_transform(df[columns])
    return df

# sample


# def detect_outliers_iqr(df, column_name):
#     data = df[column_name]
#     q25, q50, q75 = np.percentile(data, [25, 50, 75])
#     iqr = q75 - q25
#     lower_bound = q25 - 1.5 * iqr
#     upper_bound = q75 + 1.5 * iqr
#     outliers = [x for x in data if x < lower_bound or x > upper_bound]
#     outliers.sort()
#     return outliers



# # Function to detect outliers using z-score
# def detect_outliers_zscore(df, column_name):
#     data = df[column_name]
#     z_scores = np.abs(stats.zscore(data))
#     threshold = 3  # Define a threshold (e.g., 3 is commonly used)
#     outliers = [data[i] for i in range(len(data)) if z_scores[i] > threshold]
#     return outliers

# def detect_outliers_iqr(df, column_name):
#     Q1 = df[column_name].quantile(0.25)
#     Q3 = df[column_name].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)][column_name]
#     return outliers.tolist()


# def remove_outliers(df, column_name, outliers):
#     return df[~df[column_name].isin(outliers)]

# def transform_outliers(df, column_name, outliers):
#     non_outliers = df[~df[column_name].isin(outliers)]
#     median_value = non_outliers[column_name].median()
#     df.loc[df[column_name].isin(outliers), column_name] = median_value
#     return df

def detect_outliers_iqr(df, column_name):
    data = df[column_name]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
    return outliers

def remove_outliers(df, column_name, outliers):
    return df[~df[column_name].isin(outliers)]

def transform_outliers(df, column_name, outliers):
    non_outliers = df[~df[column_name].isin(outliers)]
    median_value = non_outliers[column_name].median()
    df.loc[df[column_name].isin(outliers), column_name] = median_value
    return df