#!/bin/bash 

S2U_DIR=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en

L=1024
input_file=${S2U_DIR}/train_mined_t1.09_filter${L}_u2u.tsv
output_file=${S2U_DIR}/train_mined_t1.09_filter${L}_subset0.1_u2u.tsv

# Count the number of lines in the input file (excluding the first line)
num_lines=$(tail -n +2 $input_file | wc -l)

# Calculate the number of lines to select
num_selected=$(echo "$num_lines * 0.1" | bc | awk '{print int($1+0.5)}')

# Select the lines randomly (excluding the first line), and output to a new file
tail -n +2 $input_file | shuf -n $num_selected | cat <(head -n 1 $input_file) - > $output_file
