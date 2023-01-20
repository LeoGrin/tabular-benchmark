import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor


def remove_high_cardinality(X, y, categorical_mask, threshold=20):
    high_cardinality_mask = np.array(X.nunique() > threshold)
    #TODO check
    print("masks")
    print(high_cardinality_mask)
    print(categorical_mask)
    print("high cardinality columns: {}".format(X.columns[high_cardinality_mask * categorical_mask]))
    n_high_cardinality = sum(categorical_mask * high_cardinality_mask)
    X = X.drop(X.columns[categorical_mask * high_cardinality_mask], axis=1)
    print("Removed {} high-cardinality categorical features".format(n_high_cardinality))
    categorical_mask = [categorical_mask[i] for i in range(len(categorical_mask)) if not (high_cardinality_mask[i] and categorical_mask[i])]
    return X, y, categorical_mask, n_high_cardinality




def remove_pseudo_categorical(X, y):
    """Remove columns where most values are the same"""
    pseudo_categorical_cols_mask = X.nunique() < 10
    print("Removed {} columns with pseudo-categorical values on {} columns".format(sum(pseudo_categorical_cols_mask),
                                                                                   X.shape[1]))
    X = X.drop(X.columns[pseudo_categorical_cols_mask], axis=1)
    return X, y, sum(pseudo_categorical_cols_mask)


def remove_rows_with_missing_values(X, y):
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    return X, y


def remove_missing_values(X, y, threshold=0.7, return_missing_col_mask=True):
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1]))
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    if return_missing_col_mask:
        return X, y, sum(missing_cols_mask), sum(missing_rows_mask), missing_cols_mask.values
    else:
        return X, y, sum(missing_cols_mask), sum(missing_rows_mask)


def balance(x, y):
    rng = np.random.RandomState(0)
    if len(np.unique(y)) == 1:
        # return empty arrays
        return np.array([]), np.array([])
    indices = [(y == i) for i in np.unique(y)]
    sorted_classes = np.argsort(
        list(map(sum, indices)))  # in case there are more than 2 classes, we take the two most numerous
    n_samples_min_class = sum(indices[sorted_classes[-2]])
    indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices[sorted_classes[-2]])[0]
    total_indices = np.concatenate((indices_max_class, indices_min_class))
    y = y[total_indices]
    indices_first_class = (y == sorted_classes[-1])
    indices_second_class = (y == sorted_classes[-2])
    y[indices_first_class] = 0
    y[indices_second_class] = 1

    return x.iloc[total_indices], y


def check_if_task_too_easy(X, y, categorical_indicator, regression=False, standardizer=None, max_train_samples=15000):
    train_prop = 0.7
    train_prop = min(max_train_samples / X.shape[0], train_prop)
    # TODO: only restict train set
    if X.shape[1] == 0 or X.shape[0] < 100:
        return True, pd.NA, pd.NA
    #try:
    #X = X.todense() # make dense if sparse
    #X = X.values
    #except:
    #    pass
    if standardizer is None:
        #try:
            #numeric_features = X.columns[~categorical_indicator]
            #categorical_features = X.columns[categorical_indicator]
        numeric_transformer = StandardScaler()
        numeric_transformer_sparse = MaxAbsScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
            ]
        )
        preprocessor_sparse = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer_sparse, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
            ]
        )

    print(X.shape)
    score_hbgt_list = []
    score_linear_list = []
    print(X.shape)
    if int((1 - train_prop) * X.shape[0]) > 10000:
        n_iters = 1
    elif int((1 - train_prop) * X.shape[0]) > 5000:
        n_iters = 3
    else:
        n_iters = 5
    for iter in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)

        if X_test.shape[0] > 30000:  # for speed
            indices = np.random.choice(X_test.shape[0], 30000, replace=False)
            try:
                X_test = X_test.iloc[indices]
            except: #numpy array
                X_test = X_test[indices]
            try:
                y_test = y_test.iloc[indices]
            except:
                y_test = y_test[indices]

        #try:
        #    X_train, X_test = X_train.toarray(), X_test.toarray()
        #except:
        #    print("could not convert to dense, mauybe it already is")
        #    pass
        try:
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
        except:
            print("trying MaxAbsScaler")
            X_train = preprocessor_sparse.fit_transform(X_train)
            X_test = preprocessor_sparse.transform(X_test)
        if regression:
            linear_model = TransformedTargetRegressor(LinearRegression(),
                                                      transformer=QuantileTransformer(output_distribution='normal'))
        else:
            linear_model = LogisticRegression()
        linear_model.fit(X_train, y_train)
        if regression:
            score_linear = -mean_squared_error(y_test, linear_model.predict(X_test), squared=False)  # rsme
        else:
            score_linear = linear_model.score(X_test, y_test)  # accuracy
        if regression:
            hbgt = TransformedTargetRegressor(HistGradientBoostingRegressor(max_iter=400),
                                              transformer=QuantileTransformer(output_distribution='normal'))
        else:
            hbgt = HistGradientBoostingClassifier(max_iter=400)
        hbgt.fit(X_train, y_train)
        if regression:
            score_hbgt = -mean_squared_error(y_test, hbgt.predict(X_test), squared=False)  # rsme
        else:
            score_hbgt = hbgt.score(X_test, y_test)
        score_hbgt_list.append(score_hbgt)
        score_linear_list.append(score_linear)
    print("Linear score: {}".format(score_linear_list))
    print("HBGT score: {}".format(score_hbgt_list))
    if regression:
        score_linear = np.median(score_linear_list)
        score_hbgt = np.median(score_hbgt_list)
    else:
        score_linear = np.mean(score_linear_list)
        score_hbgt = np.mean(score_hbgt_list)
    if not regression:
        if (score_hbgt - score_linear) < 0.05 * score_hbgt:
            return True, score_hbgt, score_linear
        else:
            return False, score_hbgt, score_linear
    else:
        if (score_hbgt - score_linear) < 0.05 * np.abs(score_hbgt):
            return True, score_hbgt, score_linear
        else:
            return False, score_hbgt, score_linear


def find_unwanted_columns(X, dataset_id):
    # check if dataset_id is None
    if dataset_id is None:
        return []
    if int(dataset_id) == 42571:
        return ["id"]
    elif int(dataset_id) == 42729:
        return ["id"]
    elif int(dataset_id) == 42731:
        return ["id"]
    elif int(dataset_id) == 42720:
        return ["Provider_Id", "Provider_Zip_Code"]
    elif int(dataset_id) == 43093:
        return ["PARCELNO"]
    elif int(dataset_id) == 4541:
        return ["patient_nbr"]
    else:
        return []

def specify_categorical(X, dataset_id):
    res = []
    for col_ in X.columns:
        col = str(col_).lower()
        if col.endswith("_id") or\
                col.endswith("-id") or\
                col.endswith(".id") or \
                col.endswith("_index") or \
                col.endswith("-index") or \
                col.endswith(".index"):
            res.append(col)
    if dataset_id == "42803":
        res.extend(["Vehicle_Reference_df_res",
                    "Vehicle_Type",
                    "Hit_Object_in_Carriageway",
                    "Was_Vehicle_Left_Hand_Drive?",
                    "Journey_Purpose_of_Driver",
                    "Propulsion_Code",
                    "Driver_Home_Area_Type",
                    "Accident_Severity",
                    "Local_Authority_(District)",
                    "1st_Road_Number",
                    "2nd_Road_Number",
                    "Vehicle_Reference_df",
                    "Casualty_Reference",
                    "Pedestrian_Location",
                    "Casualty_Type",
                    ])
    return res

def transform_target(y, keyword):
    if keyword == "log":
        return np.sign(y) * np.log(1 + np.abs(y))
    elif keyword == "none":
        return y
    elif pd.isnull(keyword):
        return y

