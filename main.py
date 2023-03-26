import os
os.chdir(os.path.dirname(__file__))
from scripts.analysis_pipe import main

if __name__ == '__main__':
    main()