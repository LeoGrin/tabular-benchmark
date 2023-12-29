import subprocess
import os
def count_gpu_jobs():
    #get username
    username = os.getlogin()
    print("username", username)
    # Prepare the command to get a list of nodes allocated to the user's jobs
    squeue_cmd = ['squeue', '-u', username, '--noheader', '--format=%N']

    # Run the command and capture the output
    try:
        result = subprocess.run(squeue_cmd, stdout=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running squeue: {e}")
        return 0

    # Get the list of nodes from the command output
    nodes = result.stdout.strip()
    print("nodes", nodes)
    nodes_with_gpus = [node for node in nodes if "gpu" in node]
    print("nodes_with_gpus", nodes_with_gpus)
    # Count how many node names contain 'gpu'
    gpu_count = nodes.lower().count('gpu')

    return gpu_count