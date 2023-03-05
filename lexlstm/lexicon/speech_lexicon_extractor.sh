#!/bin/bash

target=filter800_u2u

manifest_root=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests/es-en
train_u2u_file=${manifest_root}/train_mined_t1.09_${target}.tsv
lexicon_alignment_root=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/lexicon_alignment/es-en
train_lex_file=${lexicon_alignment_root}/train_mined_t1.09_${target}.tsv

mkdir -p ${lexicon_alignment_root}

if [ ! -f ${train_lex_file} ]; then
    # convert manifest u2u tsv --> fast_align tsv 
    cut -d$'\t' -f 3,5 ${train_u2u_file} | tail -n +2 | awk -F'\t' '{print $1" ||| "$2}' > ${train_lex_file}
fi

for pstr in "${target}"; do
    #echo "fast-align -i ${train_lex_file} ${p} > ${lexicon_alignment_root}/forward.align.${pstr}"
    #/data/sls/scratch/clai24/tools/fast_align/build/fast_align -i ${train_lex_file} ${p} > ${lexicon_alignment_root}/forward.align.${pstr}
    #/data/sls/scratch/clai24/tools/fast_align/build/fast_align -i ${train_lex_file} -r ${p} > ${lexicon_alignment_root}/reverse.align.${pstr}
    #/data/sls/scratch/clai24/tools/fast_align/build/atools -i ${lexicon_alignment_root}/forward.align.${pstr} -j ${lexicon_alignment_root}/reverse.align.${pstr} -c intersect >${lexicon_alignment_root}/diag.align.${pstr}
    #python summarize_aligned_data.py ${train_lex_file} ${lexicon_alignment_root}/forward.align.${pstr}
    #python summarize_aligned_data.py ${train_lex_file} ${lexicon_alignment_root}/reverse.align.${pstr}
    #python summarize_aligned_data.py ${train_lex_file} ${lexicon_alignment_root}/diag.align.${pstr}

    if [ -f ${lexicon_align_file} ]; then 
        lexicon_align_file=${lexicon_alignment_root}/diag.align.${pstr}.json
        lexicon_prob_file=${lexicon_alignment_root}/diag.align.${pstr}_probt0.1.npy
        python convert_lexicon_cnt_to_prob.py ${lexicon_align_file} ${lexicon_prob_file}
    fi 
done

