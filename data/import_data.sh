#!/bin/bash
# moves data from TSCC server to local

read -p "Enter name of directory:  " folder
scp -r "brirry@tscc-login.sdsc.edu:/home/brirry/galaxybrain/data/experiments/$folder" ./experiments

