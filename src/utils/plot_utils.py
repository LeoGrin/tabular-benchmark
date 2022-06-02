import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_prefix_to_keys(dic, prefix):
    new_dic = {}
    for key in dic.keys():
        new_dic["{}__{}".format(prefix, key)] = dic[key]
    return new_dic


def plot_decision_boudaries(X_train, y_train, X_test, y_test, clf, title="decision boundaries", x_min=None, x_max=None,
                            y_min=None, y_max=None):
    # Code source: Gaël Varoquaux
    #              Andreas Müller
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots()
    h = 0.02  # step size in the mesh

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    if x_min is None:
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)


    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    # Plot the training points
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", s=1)
    # Plot the testing points
    #  plt.scatter(
    #     X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k", s=3
    # )
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.set_xticks(())
    #ax.set_yticks(())

    # iterate over classifiers
    # clf.fit(X_train, y_train)
    #score = clf.score(X_test, y_test)
    score = clf.score(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

    # Plot the training points
    ax.scatter(
        # X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, s=6, edgecolors="k", linewidth=0.2
    )
    # # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        edgecolors="k",
        alpha=1,
        marker="^",
        s=6,
        linewidth=0.2
    )

    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.set_xticks(())
    #ax.set_yticks(())

    ax.text(
        x_min - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_comparison(df, data_dic, target_dic, transform_dic_list, model_names):
    print(data_dic)
    print(target_dic)
    steps = ["data", "target"]
    print(len(df))
    for i, dic in enumerate([data_dic, target_dic]):
        df = df[np.logical_and.reduce(
            [((df[k] == v) | (pd.isna(df[k]) & pd.isna(v))) for k, v in add_prefix_to_keys(dic, steps[i]).items()])]
        print(len(df))
    for i, transform_dic in enumerate(transform_dic_list):
        print("here")
        print(transform_dic)
        print(add_prefix_to_keys(transform_dic, "transform__" + str(i)).items())
        df = df[np.logical_and.reduce([((df[k] == v) | (pd.isna(df[k]) & pd.isna(v))) for k, v in
                                       add_prefix_to_keys(transform_dic, "transform__" + str(i)).items()])]
        print(len(df))

    if df.groupby('model_name').count().max() == 1:
        n_iter = df["n_iter"].item()
        test_scores_mean = np.array([df["test_scores_{}_mean".format(name)].item() for name in model_names])
        test_scores_sd = np.array([df["test_scores_{}_sd".format(name)].item() for name in model_names])
        train_scores_mean = np.array([df["train_scores_{}_mean".format(name)].item() for name in model_names])
        train_scores_sd = np.array([df["train_scores_{}_sd".format(name)].item() for name in model_names])
        test_bottom = min(
            np.array([test_scores_mean[i] - 3 * test_scores_sd[i] / np.sqrt(n_iter) for i in range(len(model_names))]))
        train_bottom = min(np.array(
            [train_scores_mean[i] - 3 * train_scores_sd[i] / np.sqrt(n_iter) for i in range(len(model_names))]))
        fig, ax = plt.subplots(2)

        ticks = np.arange(len(model_names))
        ax[0].bar(ticks, test_scores_mean - test_bottom,
                  yerr=2 * test_scores_sd / np.sqrt(n_iter),
                  bottom=test_bottom)
        ax[0].pltes.xticks(ticks)
        ax[0].pltes.set_xticklabels(model_names)
        ax[0].set_title("Test scores")
        ax[1].bar(ticks, train_scores_mean - train_bottom,
                  yerr=2 * train_scores_sd / np.sqrt(n_iter),
                  bottom=train_bottom)
        ax[1].pltes.xticks(ticks)
        ax[1].pltes.set_xticklabels(model_names)
        ax[1].set_title("Train scores")
        fig.tight_layout()
        return fig
    elif df.groupby('model_name').count().max() > 1:
        print(df)
        raise ValueError("Too many possibilities")
    elif df.groupby('model_name').count().max() < 1:
        raise ValueError("Results not saved")


if __name__ == '__main__':
    df = pd.read_csv('../../results/old/clean_results_15_10.csv')
    data_generation_dic = {"method_name": "gaussian",
                           "num_samples": 1000,
                           "num_features": 15,
                           "cov_matrix": "random"}
    target_generation_dic = {"method_name": "random_forest",
                             "n_trees": 20,
                             "max_depth": 5,
                             "depth_distribution": "uniform",
                             "split_distribution": "uniform"}
    data_transforms_dic = {"method_name": "add_noise",
                           "noise_type": "white",
                           "scale": 0.1}
    fig = plot_comparison(df,
                          data_generation_dic,
                          target_generation_dic,
                          data_transforms_dic,
                          ["mlp_skorch", "gbt", "rf"])
