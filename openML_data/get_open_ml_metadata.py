import openml
import pandas as pd

l = []
for task_id in openml.study.get_suite('218').tasks:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    dic = dataset.qualities
    dic["id"] = task_id
    l.append(dic)

df = pd.DataFrame(l)

df.to_csv("open_ml_attributes_218.csv")