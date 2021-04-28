# moves data in this directory to TSCC
for i in {1..6}
do 
    scp -r experiments/exp$i brirry@tscc-login.sdsc.edu:../../projects/ps-voyteklab/brirry/galaxybrain_data/experiments
done
