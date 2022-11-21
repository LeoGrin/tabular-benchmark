# Run a model to compute attributions and compare them to a baseline.
import numpy as np
import matplotlib.pyplot as plt
from utils.parser import get_attribution_parser
from models.TabSurvey.models import str2model
from utils.load_data import load_data
from utils.io_utils import save_results_to_json_file
from sklearn.model_selection import train_test_split
from utils.baseline_attributions import get_shap_attributions
from models.TabSurvey.models import BaseModel
import typing as tp


def train_model(args, model: BaseModel, X_train: np.ndarray, X_val: np.ndarray,
                y_train: np.ndarray, y_val: np.ndarray) -> BaseModel:
    """ Train model using parameters args. 
        X_train, y_train: Training data and labels
        X_val and y_val: Test data and 
        :return: Trained model.
    """
    loss_history, val_loss_history = model.fit(X_train, y_train, X_val, y_val)
    val_model(model, X_val, y_val)
    return model


def global_removal_benchmark(args, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                             y_val: np.ndarray, feature_importances: np.ndarray, order_morf=True) -> np.ndarray:
    """ Perform a feature removal benchmark for the attributions. 
        The features that are attributed the highest overall attribution scores are successivly removed from the 
        dataset. The model is then retrained.
        
        :param features_importances: A vector of D (number of features in X) values that contain the importance score for each feature.
            The features will be ordered by the absolute value of the passed importance.
        :param X_val: (N, D) train data (N samples, D features)
        :param y_val: (N) train class labels
        :param X_val: (M, D) test data (M samples, D features)
        :param y_val: (M) test class labels
        :param order_morf: Feature removal order. Either remove most important (morf=True) or least important (morf=False) features first
        :return: array with the obtained accuracies.
    """
    if X_train.shape[1] != len(feature_importances):
        raise ValueError("Number of Features in Trainset must be equal to number of importances passed.")

    ranking = np.argsort((1 if order_morf else -1) * np.abs(feature_importances))
    results = np.zeros(len(feature_importances))
    old_cat_index = args.cat_idx
    old_cat_dims = args.cat_dims
    for i in range(len(feature_importances)):
        remaining_features = len(feature_importances) - i
        use_idx = ranking[:remaining_features].copy()
        np.random.shuffle(use_idx)  # make sure the neighborhood relation is not important.

        print(f"Using {len(use_idx)} features ...")
        # Retrain the model and report acc.
        X_train_bench = X_train[:, use_idx]
        X_val_bench = X_val[:, use_idx]

        # modify feature args accordingly
        # args.num_features: points to the new number of features
        # args.cat_idx: Indices of categorical features
        # args.cat_dims: Number of categorical feature values
        # These values have to be recomputed for the modified dataset
        new_cat_idx = []
        new_cat_dims = []
        for j in range(len(use_idx)):
            if use_idx[j] in old_cat_index:
                old_index = old_cat_index.index(use_idx[j])
                new_cat_idx.append(j)
                new_cat_dims.append(old_cat_dims[old_index])

        args.cat_idx = new_cat_idx
        args.cat_dims = new_cat_dims
        args.num_features = remaining_features

        model_name = str2model(args.model_name)
        model = model_name(arguments.parameters[args.model_name], args)
        model = train_model(args, model, X_train_bench, X_val_bench, y_train, y_val)
        acc_obtained = val_model(model, X_val_bench, y_val)
        results[i] = acc_obtained

        res_dict = {}
        res_dict["model"] = args.model_name
        res_dict["order"] = "MoRF" if order_morf else "LeRF"
        res_dict["accuracies"] = results.tolist()
        res_dict["attributions"] = feature_importances.tolist()
    save_results_to_json_file(args, res_dict, f"global_benchmark{args.strategy}", append=True)
    # reset args to their old values.
    args.cat_idx = old_cat_index
    args.cat_dims = old_cat_dims
    return results


def compute_spearman_corr(attr1: np.ndarray, attr2: np.ndarray) -> np.ndarray:
    """ Compute the spearman rank correlations between two attributions. The attributions are first ranked 
        by their value. Pass absolute values, if you want to rank by magnitude only.
        Return a vector with the spearman correlation between all rows in the matrix.
        :param attr1: (N, D) attributions by method 1 (N samples, D features)
        :param attr2: (N, D) attributions by method 2 (N samples, D features)
        :return: (N) array with the rank correlation of the two attributions for each sample.
    """
    num_inputs = attr1.shape[0]
    resmat = np.zeros(num_inputs)
    ranks1 = np.argsort(np.argsort(attr1, axis=0), axis=0)
    ranks2 = np.argsort(np.argsort(attr2, axis=0), axis=0)

    cov = np.mean(ranks1 * ranks2, axis=0) - np.mean(ranks1, axis=0) * np.mean(ranks2, axis=0)  # E[XY]-E[Y]E[X]
    corr = cov / (np.std(ranks1, axis=0) * np.std(ranks2, axis=0))
    return corr


def compare_to_shap(args, attrs, model, X_val, sample_size=100):
    """ 
        Compare feature attributions by the model to shap values on a random set of validation points.
        Compute correlation and save raw output to JSON file.
        :param attrs: (N, D) model feature attributions
        :param model: The model to use.
        :param X_val: (N, D) test data (N samples, D features)
        :param sample_size: Number of points to choose
    """
    use_samples = np.arange(len(X_val))
    np.random.shuffle(use_samples)
    use_samples = use_samples[:sample_size]
    attrs = attrs[use_samples]

    res_dict = {}
    res_dict["model"] = args.model_name
    res_dict["model_attributions"] = attrs.tolist()

    shap_attrs = get_shap_attributions(model, X_val[use_samples])
    # save_attributions_image(attrs, feature_names, args.model_name+"_shap")
    res_dict["shap_attributions"] = shap_attrs.tolist()

    rank_corrs = compute_spearman_corr(np.abs(attrs), np.abs(shap_attrs))
    res_dict["rank_corr_mean"] = np.mean(rank_corrs)
    res_dict["rank_corr_std"] = np.std(rank_corrs)
    save_results_to_json_file(args, res_dict, f"shap_compare{args.strategy}", append=True)


def val_model(model: BaseModel, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """ 
        Validation of a trained classification model on the test set (X_val, y_val). 
        :param X_val: (N, D) test data (N samples, D features)
        :param y_val: (N) test class labels
        :return: accuracy
    """
    ypred = model.predict(X_val)
    if len(ypred.shape) == 2:
        ypred = ypred[:, -1]
    acc = np.sum((ypred.flatten() > 0.5) == y_val) / len(y_val)
    print("Accuracy: ", acc)
    return acc


def save_attributions_image(attrs: np.ndarray, namelist: tp.Optional[tp.List[str]] = None,
                            file_name: str = ""):
    """ Save attributions in a plot. 
        :param attrs: (N, D) attributions (N samples, D features)
        :param namelist: List of length D with column names
        :return: predicted labels of test data
    """
    attrs_abs = np.abs(attrs)
    attrs_abs -= np.min(attrs_abs)
    attrs_abs /= np.max(attrs_abs)
    plt.ioff()
    plt.matshow(attrs_abs)
    if namelist:
        plt.xticks(np.arange(len(namelist)), namelist, rotation=90)
    plt.tight_layout()
    plt.gcf().savefig(f"output/attributions_{file_name}.png")


def main(args):
    if args.model_name == "TabTransformer":  # Use discretized version of adult dataset for TabNet attributions.
        args.scale = False

    # Load dataset (currently only tested for the Adult data set)
    X, y = load_data(args)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=args.seed)

    # Create the model
    model_name = str2model(args.model_name)
    model = model_name(arguments.parameters[args.model_name], args)
    # Obtain a trained model to get attributions
    modelref = train_model(args, model, X_train, X_val, y_train, y_val)
    # Get attributions
    attrs = modelref.attribute(X_val, y_val, args.strategy)

    # Save the first 20 attributions to file.
    if args.dataset == "Adult" or args.dataset == "AdultCat":
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country']
    else:
        feature_names = None
    res_dict = {}
    res_dict["model"] = args.model_name
    res_dict["strategy"] = str(args.strategy)
    res_dict["dataset"] = args.dataset
    res_dict["attributions"] = attrs.tolist()
    save_results_to_json_file(args, res_dict, f"attributions{args.strategy}", append=True)
    save_attributions_image(attrs[:20, :], feature_names, args.model_name)

    # Run global attribution benchmark if flag is passed.
    if args.globalbenchmark:
        for order in [True, False]:
            for run in range(args.numruns):
                global_removal_benchmark(args, X_train, X_val, y_train, y_val, attrs.mean(axis=0).flatten(),
                                         order_morf=order)

    # Compute Shaples values and compare to model intrinsic attribution if flag is passed.
    if args.compareshap:
        compare_to_shap(args, attrs, modelref, X_val, sample_size=250)


if __name__ == "__main__":
    parser = get_attribution_parser()
    arguments = parser.parse_args()
    main(arguments)
