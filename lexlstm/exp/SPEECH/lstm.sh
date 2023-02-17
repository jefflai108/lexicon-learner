#!/bin/bash
CUDA_VISIBLE_DEVICES=5
max_steps=30000
expname=en-es/1000/LSTM/
mkdir -p $expname
# cd $expname
home="/home/akyurek/git/lexicon-learner/lexlstm/"
cd $home
cnt=0
for lr in 0.5; do
    for warmup_steps in 8000; do
        for p_augmentation in 0.0; do
            for seed in 0; do
                subexpname=${home}/exp/SPEECH/${expname}/lr_${lr}_ws_${warmup_steps}/seed_${seed}/
                mkdir -p ${subexpname}
                python -u  $home/main.py \
                    --seed $seed \
                    --exp_name="SPEECH/LSTM/seed_${seed}/" \
                    --n_batch 512 \
                    --n_layers 2 \
                    --dim 512 \
                    --lr=${lr} \
                    --temp 1.0 \
                    --dropout=0.4 \
                    --beam_size=5 \
                    --gclip=5.0 \
                    --accum_count=1 \
                    --valid_steps=2000 \
                    --warmup_steps=${warmup_steps} \
                    --max_step=${max_steps} \
                    --tolarance=10 \
                    --tb_dir=${subexpname} \
                    --p_augmentation=${p_augmentation} \
                    --gpu=$(( cnt % 2 )) \
                    --dataset="speech" \
                    --copy > ${subexpname}/eval.out 2> ${subexpname}/eval.err &
                cnt=$(( cnt + 1 ))
            done
        done
    done
done
