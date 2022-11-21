import os
import errno
import scipy.io
import shutil
import json
from os.path import isfile, join, isdir


def dir_to_file_list(path_to_dir):
    return [path_to_dir + '/' + f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]


def dir_to_file_names_list(path_to_dir):
    return [f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]


def dir_to_file_list_with_ext(path_to_dir, ext):
    return list(filter(lambda x: x.endswith(ext), dir_to_file_list(path_to_dir)))


def dir_to_subdir_list(path_to_dir):
    return [path_to_dir + '/' + f for f in os.listdir(path_to_dir) if isdir(join(path_to_dir, f))]


def create_dir(path):
    if os.path.exists(path):
        return

    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def delete_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def delete_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def get_file_name(path):
    path = path.replace('\\', '/')
    return path.split('/')[-1]


def get_file_name_without_ext(path):
    name_with_ext = get_file_name(path)
    return name_with_ext.split(".")[0]


def read_list_from_file(file_path):
    with open(file_path, "r") as file:
        l = []
        for line in file:
            l.append(line.rstrip())
        return l


def file_to_path_list(file_path):
    with open(file_path, "r") as file:
        l = []
        for line in file:
            l.append(line.rstrip())
    return l


def write_list_to_file(data_list, file_path):
    with open(file_path, "a") as file:
        for item in data_list:
            file.write(str(item) + "\n")


def write_dict_to_file(data_dict, file_path):
    with open(file_path, 'w') as file:
        file.write(json.dumps(data_dict,  sort_keys=False, indent=4, separators=(',', ': ')))


def read_dict_from_file(file_path):
    with open(file_path) as f:
        return json.load(f)


def sort_files(file_list):
    file_list = [(int(get_file_name_without_ext(x)), x) for x in file_list]
    file_list = sorted(file_list, key=lambda x: x[0])
    file_list = [x[1] for x in file_list]
    return file_list


def read_mat(file):
    return scipy.io.loadmat(file)


def write_mat(file, mat):
    '''
    :param mat: dictionary of names and arrays
    '''
    return scipy.io.savemat(file, mat)