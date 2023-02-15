exp_root="../SPEECH/"
train_file=${exp_root}/train_mined_t1.09_ekin.filter1000.fast

# if [ ! -f ${train_file} ]; then
cut -d $'\t' -f 1-2 ${exp_root}/train_mined_t1.09_ekin.filter1000.tsv  | sed 's/\t/ ||| /g' > ${train_file}
# fi

alignment_root="${exp_root}/align_lexsym_w10_intersection_log/"
mkdir -p ${alignment_root}
for p in "-o"; do
    pstr=${p//-/}
    echo "fast_align -i ${train_file} ${p} > ${alignment_root}/forward.align.${pstr}"
    fast_align -i ${train_file} ${p} > ${alignment_root}/forward.align.${pstr}
    fast_align -i ${train_file} -r ${p} > ${alignment_root}/reverse.align.${pstr}
    atools -i ${alignment_root}/forward.align.${pstr} -j ${alignment_root}/reverse.align.${pstr} -c intersect >${alignment_root}/diag.align.${pstr}
    python summarize_aligned_data.py ${train_file} ${alignment_root}/forward.align.${pstr}
    python summarize_aligned_data.py ${train_file} ${alignment_root}/reverse.align.${pstr}
    python summarize_aligned_data.py ${train_file} ${alignment_root}/diag.align.${pstr}
    mkdir -p ${alignment_root}/logs/
done
