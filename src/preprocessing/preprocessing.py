from sklearn.preprocessing import LabelEncoder
from preprocessing.utils import *
import numpy as np


def preprocessing(X, y, categorical_indicator, categorical, regression, transformation=None, dataset_id=None):
    original_n_samples, original_n_features = X.shape
    le = LabelEncoder()
    if not regression:
        y = le.fit_transform(y)
    binary_variables_mask = np.array(X.nunique() == 2)
    print(categorical_indicator)
    for i in range(X.shape[1]):
        if binary_variables_mask[i]:
            categorical_indicator[i] = True
    for i in range(X.shape[1]):
        if type(X.iloc[1, i]) == str:
            categorical_indicator[i] = True

    if not dataset_id is None:
        # Returns the list of specific categorical variables not indicated as categorical
        specific_categorical = specify_categorical(X, dataset_id)
        for i in range(X.shape[1]):
            if X.columns[i] in specific_categorical:
                categorical_indicator[i] = True

    num_categorical_columns = sum(categorical_indicator)
    print("Number of categorical columns: {}".format(num_categorical_columns))

    pseudo_categorical_mask = np.array(X.nunique() < 10)
    n_pseudo_categorical = 0
    cols_to_delete = []
    for i in range(X.shape[1]):
        if pseudo_categorical_mask[i]:
            if not categorical_indicator[i]:
                n_pseudo_categorical += 1
                cols_to_delete.append(i)
    print("Number of pseudo categorical variables: {}".format(n_pseudo_categorical))
    if not categorical:
        for i in range(X.shape[1]):
            if categorical_indicator[i]:
                cols_to_delete.append(i)

    if not dataset_id is None:
        unwanted_columns = find_unwanted_columns(X, dataset_id)
        print("Number of unwanted columns: {}".format(len(unwanted_columns)))
        for i in range(X.shape[1]):
            if X.columns[i] in unwanted_columns:
                cols_to_delete.append(i)
    print("cols to delete")
    print(X.columns[cols_to_delete])
    print("{} columns removed".format(len(cols_to_delete)))
    X = X.drop(X.columns[cols_to_delete], axis=1)
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                             not i in cols_to_delete]  # update categorical indicator

    X, y, categorical_indicator, num_high_cardinality = remove_high_cardinality(X, y, categorical_indicator, 20)
    print("Remaining categorical")
    print([X.columns[i] for i in range(X.shape[1]) if categorical_indicator[i]])
    X, y, num_columns_missing, num_rows_missing, missing_cols_mask = remove_missing_values(X, y)
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                             not missing_cols_mask[i]]

    if X.shape[0] > 1:
        if not regression:
            X, y = balance(X, y)
            y = le.fit_transform(y)
            # assert len(X) == len(y)
            assert len(np.unique(y)) == 2
            assert np.max(y) == 1
        for i in range(X.shape[1]):
            if categorical_indicator[i]:
                X.iloc[:, i] = LabelEncoder().fit_transform(X.iloc[:, i])

        if transformation is not None and transformation != "none":
            assert regression
            y = transform_target(y, transformation)
        else:
            print("NO TRANSFORMATION")

    return X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, num_categorical_columns, \
           n_pseudo_categorical, original_n_samples, original_n_features
