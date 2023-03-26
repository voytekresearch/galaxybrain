import sys
import subprocess
if 'sdsc' in subprocess.run('hostname', capture_output=True).stdout.decode('utf8'):
    sys.path.append('/home/brirry/galaxybrain')
from scripts.analysis_pipe import main

if __name__ == '__main__':
    main()