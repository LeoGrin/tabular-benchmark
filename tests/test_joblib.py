from joblib import Parallel, delayed, effective_n_jobs, cpu_count
import joblib
import time

from dask_jobqueue import OARCluster
cluster = OARCluster(cores=1, memory='1G', walltime='10:00:00')
cluster.scale(jobs=100)  # ask for 10 jobs
from dask.distributed import Client
client = Client(cluster)

def foo(i):
    time.sleep(30)
    return i

print("begin")
print(time.time())
start_time = time.time()
print(effective_n_jobs())
print(cpu_count())

with joblib.parallel_backend('dask'):
    joblib.Parallel(verbose=100)(
        joblib.delayed(foo)(i)
        for i in range(100))