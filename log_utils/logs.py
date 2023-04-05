import logging
import subprocess as sp


def init_log(node):
    name = f'analysis_node{node}'
    host = sp.check_output('hostname', shell=True).decode('utf-8').lower()
    if 'brian' in host:
        log_path = f'/Users/brianbarry/Desktop/computing/galaxybrain/logs/{name}.log'
    elif 'tscc' in host:
        log_path = f'/home/brirry/logs/{name}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s at %(filename)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger