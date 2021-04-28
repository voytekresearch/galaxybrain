# moves data from TSCC server to local

read -p "Enter name of directory:  " folder
scp -r "brirry@tscc-login.sdsc.edu:../../projects/ps-voyteklab/brirry/data/experiments/$folder" ./experiments

