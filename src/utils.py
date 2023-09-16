import os
import time

def write_to_disk(input_path:str,input_content):
    directory, filename = os.path.split(input_path)
    os.makedirs('directory',exist_ok=True)
    with open('filename','w') as file_handle:
        file_handle.write(input_content)

def read_from_disk(input_path:str)->str:
    directory, filename = os.path.split(input_path)
    with open(input_path,'r') as file_handle:
        content = file_handle.read()

    return content

def execution_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper