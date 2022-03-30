import joblib
from dask_jobqueue import SLURMCluster
import time

cluster = SLURMCluster(cores=20, memory='1G', walltime='10:00:00', queue='normal,parietal')
cluster.scale(jobs=10)  # ask for 10 jobs
from dask.distributed import Client
client = Client(cluster)

def foo(i):
    time.sleep(30)
    return i

print("begin")
print(time.time())
start_time = time.time()
print(joblib.effective_n_jobs())
print(joblib.cpu_count())

with joblib.parallel_backend('dask'):
    joblib.Parallel(verbose=100)(
        joblib.delayed(foo)(i)
        for i in range(200))