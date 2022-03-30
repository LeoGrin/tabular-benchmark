import joblib
from dask_jobqueue import SLURMCluster
import time
import torch

cluster = SLURMCluster(cores=1,
                       # processes=1,
                       memory='100GB',
                       walltime='10:00:00',
                       job_name="test_joblib_cuda",
                       queue="jazzy",
                       job_extra=[f'--gres=gpu:1'])
cluster.scale(jobs=2)  # ask for 10 jobs
from dask.distributed import Client
client = Client(cluster)

def foo(i):
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(torch.cuda.current_device()))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    #time.sleep(5)
    return i

#print("begin")
#print(time.time())
#start_time = time.time()

print(cluster.job_script())

with joblib.parallel_backend('dask'):
    joblib.Parallel(verbose=100)(
        joblib.delayed(foo)(i)
        for i in range(30))