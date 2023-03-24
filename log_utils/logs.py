import logging
import subprocess as sp


def init_log():
    host = sp.check_output('hostname', shell=True).decode('utf-8').lower()
    if 'brian' in host:
        log_path = '/Users/brianbarry/Desktop/computing/galaxybrain/logs/analysis.log'
    elif 'tscc' in host:
        log_path = '/home/brirry/logs/analysis.log'
    level = logging.DEBUG

    logging.basicConfig(filename=log_path,
                        level=level,
                        format='%(asctime)s | %(levelname)s at %(filename)s: %(message)s')