import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv("datasets/categorical_regression.csv")

columns_criteria = ["High dimensional", "Stream", "Time series", "Too many low cardinality", "Hidden columns", "Too little information", "Too many IDs",	"Redundant", "Artificial", "Not Heterogeneous", "Deterministic"]

df["Not Heterogeneous"] = df["heterogeneous"] == False



link_dict = {}

label_dic = {
    "Test": 0,
    "> 50K samples": 1,
    "< 50K samples": 2,
    "Removing NA": 3,
    "Balancing": 4,
    "Checks": 5,
    "Passed": 6,
    "Used": 7,
    "Too easy": 8,
    "Not too easy": 9,
    "First Passed": 10,
    "Too small": 11,
    "Not used": 12,
}

cutoff = 3_000

df = df[((df["original_n_samples"] > cutoff) & (df["n_features"] > 3)) | (pd.isnull(df["original_n_features"]))]

source = []
target = []
value = []

source.append(label_dic["Removing NA"]),
target.append(label_dic["Balancing"]),
value.append(((df["original_n_samples"] - df["num_rows_missing"] >= cutoff) | (pd.isnull(df["original_n_samples"]))).sum())

source.append(label_dic["Removing NA"])
target.append(label_dic["Too small"])
value.append((df["original_n_samples"] - df["num_rows_missing"] < cutoff).sum())

too_small_na = (df["original_n_samples"] - df["num_rows_missing"] < cutoff).sum()

df = df[(df["original_n_samples"] - df["num_rows_missing"] >= cutoff) | (pd.isnull(df["original_n_samples"]))]

source.append(label_dic["Balancing"])
target.append(label_dic["Too small"])
value.append((df["n_samples"] < cutoff).sum())

too_small_balancing = (df["n_samples"] < cutoff).sum()

source.append(label_dic["Too small"])
target.append(label_dic["Not used"])
value.append(too_small_balancing + too_small_na)


source.append(label_dic["Balancing"])
target.append(label_dic["Checks"])
value.append(((df["n_samples"] >= cutoff) | (pd.isnull(df["original_n_samples"]))).sum())



df = df[(df["n_samples"] >= cutoff) | (pd.isnull(df["original_n_samples"]))]

if "n_features_probable" in df.columns:
    df["Too few cat cols"] = np.abs(df["n_features"] - df["n_features_probable"]) / df["n_features"] < 0.1
    columns_criteria.append("Too few cat cols")

if "ratio_n_p" in df.columns:
    df["High dimensional"] = 1 - df["ratio_n_p"]
else:
    df["High dimensional"] = df["P/n ratio too high"]


#
#
offset = len(label_dic.keys())
for i, col in enumerate(columns_criteria):
    if col in df.columns:
        idx = i + offset
        label_dic[col] = idx
        num_rejected = df[df[col] == True].shape[0]
        df = df[~(df[col] == True)]

        # Link rejected > 50K samples to the criterion col
        source.append(label_dic["Checks"])
        target.append(idx)
        value.append(num_rejected)
        source.append(idx)
        target.append(label_dic["Not used"])
        value.append(num_rejected)
        # Link not rejected to "Too easy"
        #df[df[col] == True]["too_easy"] = True
        # Link not rejected to "First Passed"
    else:
        offset -= 1


# # Link Firt Passed to "Too easy" and "Large dataset"
# print("-" * 80)
# print(len(df))
# print(df["too_easy"].sum())
source.append(label_dic["Checks"])
target.append(label_dic["Passed"])
value.append(len(df))
source.append(label_dic["Passed"])
target.append(label_dic["Used"])
value.append(len(df[~(df["too_easy"] == True)]))
source.append(label_dic["Passed"])
target.append(label_dic["Too easy"])
value.append(len(df[(df["too_easy"] == True)]))
source.append(label_dic["Too easy"])
target.append(label_dic["Not used"])
value.append(len(df[(df["too_easy"] == True)]))


print(list(label_dic.keys()))
print(dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value
  ))

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 30,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = list(label_dic.keys()),
      color = ["red" if (key in columns_criteria or key == "Too small" or key == "Too easy" or key == "Not used") else "blue" for key in label_dic.keys()]
    ),
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value
  ))])

fig.update_layout(title_text="", font_size=25)
fig.write_image("plots/sankey_categorical_regression.png", width=1200, scale=5)