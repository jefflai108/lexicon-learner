import re, glob, os, string
import csv
from tqdm import tqdm
import textgrid
import pickle

def read_textgrid(file, seg_type='words'): 
    # Load TextGrid file
    tg = textgrid.TextGrid.fromFile(file)

    # Get tier with specific name
    tier_name = seg_type 
    tier = tg.getFirst(tier_name)

    # Extract words and their start/end times as tuples
    word_tups = []
    for interval in tier.intervals:
        word = interval.mark
        start_time = interval.minTime
        end_time = interval.maxTime
        word_tups.append((word, start_time, end_time))

    return word_tups

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser('Post-process textgrid files')

    parser.add_argument('--seg_type', type=str, default='phones')
    parser.add_argument('--split', type=str, default='en-test_epst')
    parser.add_argument('--lan_pair', type=str, default='s2u_en-es')
    parser.add_argument('--save_root', type=str, default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/mfa_s2u_manifests/')
    parser.add_argument('--data_root', type=str, default='/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/')
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.lan_pair, f'{args.split}.tsv'), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)[0]  # skip the header row
        wavs = [row[0] for row in reader] 
    
    print(wavs)
    print(len(wavs))
    # read .tsv files instead. 
    # if there's no TextGrid file, set alignment to []

    # Iterate through all the text files in the directory
    no_alignment = 0
    utt2seg = {}
    for wav in tqdm(wavs): 
        # first softlink audio to target dir
        uttid = wav.split('/')[-1]
        if uttid.endswith('.flac'):
            uttid = uttid.strip('.flac')
        elif uttid.endswith('.wav'):
            uttid = uttid.strip('.wav')

        if args.split == 'en-valid_vp' or args.split == 'en-train_mined_t1.09':
            tmp_uttid = wav.replace(':', '-').replace('.zip', '')
        elif args.split == 'en-test_fleurs': 
            tmp_uttid = uttid + '.'
        else: 
            tmp_uttid = uttid
        
        print(uttid)
        textgird_fpth = os.path.join(args.save_root, args.lan_pair, f"{args.split}-aligned", f"{tmp_uttid}.TextGrid")
        if os.path.exists(textgird_fpth): 
            tuples = read_textgrid(textgird_fpth, args.seg_type)
        else: 
            no_alignment += 1 
            tuples = [(None, 0.0, 0.0)]

        utt2seg[uttid] = tuples
        print(textgird_fpth)

    with open(os.path.join(args.save_root, args.lan_pair, f"{args.split}-{args.seg_type}_seg.pkl"), 'wb') as f:
        pickle.dump(utt2seg, f)

    print(f"there are {no_alignment} no alignments")

