import whisper 
import os, csv
from tqdm import tqdm 
import soundfile as sf

from fairseq_extract_waveform import get_features_or_waveform

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser('Transcribe tsv file')

    parser.add_argument('--split', type=str, default='en-valid_vp')
    parser.add_argument('--lan_pair', type=str, default='s2u_en-es')
    parser.add_argument('--data_root', type=str, default='/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/')
    parser.add_argument('--save_root', type=str, default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/mfa_s2u_manifests/')
    parser.add_argument('--nprocess', type=int, default=1) 
    parser.add_argument('--process_id', type=int, default=0) 
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.lan_pair, f'{args.split}.tsv'), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)[0]  # skip the header row
        wavs = [row[0] for row in reader] 
        if args.nprocess > 1: 
            increment = len(wavs) // args.nprocess 
            if args.process_id > args.nprocess: 
                print(f"invalid spcification of args.process_id")
                exit()
            new_wavs = wavs[args.process_id * increment: (args.process_id+1) * increment]
            wavs = new_wavs

    model = whisper.load_model("large")
    
    for wav in tqdm(wavs): 
        uttid = wav.replace(':', '-').replace('.zip', '')
        target_ln = os.path.join(args.save_root, args.lan_pair, args.split, f"{uttid}.wav")
        feat = get_features_or_waveform(os.path.join(header, wav), need_waveform=True, use_sample_rate=16000) 

        sf.write(target_ln, feat, 16000)

        # transcribe with Whisper 
        result = model.transcribe(target_ln)
        
        # write to output 
        target_f = os.path.join(args.save_root, args.lan_pair, args.split, f"{uttid}.txt")
        with open(target_f, 'w') as f: 
            f.write('%s\n' % result['text'])

