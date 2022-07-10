from dirty_cat.datasets import *

fetch_road_safety().X.to_csv("dirty_cat_temp/road_safety.csv")

fetch_midwest_survey().X.to_csv("dirty_cat_temp/midwest.csv")

fetch_medical_charge().X.to_csv("dirty_cat_temp/medical_charge.csv")

fetch_open_payments().X.to_csv("dirty_cat_temp/open_payments.csv")

fetch_traffic_violations().X.to_csv("dirty_cat_temp/traffic.csv")

fetch_drug_directory().X.to_csv("dirty_cat_temp/drug.csv")



