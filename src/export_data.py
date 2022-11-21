import pandas as pd
import os
os.environ["PROJECT_DIR"] = "test"
import openml
print("ho")
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(".")
import argparse
import numpy as np


# Create an argument parser
parser = argparse.ArgumentParser(description='Train a model on a dataset')

# Add the arguments
parser.add_argument('--device', type=str, default="cuda", help='Device to use')
parser.add_argument('--file', type=str, default="filename", help='Csv with all datasets')
parser.add_argument('--out_file', type=str, default="filename", help='filename to save')
# true if argument is present, false otherwise
parser.add_argument('--regression', action='store_true', help='True if regression, false otherwise')
parser.add_argument('--categorical', action='store_true')
# Parse the arguments
args = parser.parse_args()


df = pd.read_csv("data/aggregates/{}.csv".format(args.file))





res_df = pd.DataFrame()


for index, row in df.iterrows():
    #try:
    if not pd.isnull(row["dataset_id"]) and row["Remove"] != 1 and row["too_easy"] == 1 and row["Redundant"] != 1:
        prefix_to_skip = ["BNG", "RandomRBF", "GTSRB", "CovPokElec", "PCam"]
        if not(np.any([row["dataset_name"].startswith(prefix) for prefix in
                   prefix_to_skip]) or "mnist" in row["dataset_name"].lower() or "image" in row["dataset_name"].lower() or "cifar" in row["dataset_name"].lower() or row["dataset_id"] == 1414):
            print(row["dataset_name"])
            os.system("scp -r /Users/leo/.openml/org/openml/www/datasets/{} drago:/storage/store/work/lgrinszt/org/openml/www/datasets/".format(int(row["dataset_id"])))