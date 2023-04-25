import whisper 
from num2words import num2words
import os, csv
from tqdm import tqdm 

#model = whisper.load_model("large")
#result = model.transcribe("/data/sls/temp/clai24/data/speech_matrix/eval_data/europarl_st/fairseq_processed/audios/en/en.20110608.24.3-557-000_68.57_71.59.wav")
#print(result["text"])

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser('Transcribe tsv file')

    parser.add_argument('--split', type=str, default='en-test_fleurs')
    parser.add_argument('--lan_pair', type=str, default='s2u_en-es')
    parser.add_argument('--data_root', type=str, default='/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/')
    parser.add_argument('--save_root', type=str, default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/mfa_s2u_manifests/')
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.lan_pair, f'{args.split}.tsv'), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)[0]  # skip the header row
        wavs = [row[0] for row in reader] 

    model = whisper.load_model("large")

    for wav in tqdm(wavs): 
        # first softlink audio to target dir
        uttid = wav.split('/')[-1][:-4]
        target_ln = os.path.join(args.save_root, args.lan_pair, args.split, f"{uttid}.wav")
        if args.split in ['en-test_fleurs', 'en-test_epst', 'es_dummy']:
            if args.split == 'es_dummy':
                wav = os.path.join(header, wav)
            if not os.path.lexists(target_ln):
                os.symlink(wav, target_ln)
        else: 
            uttid = wav.replace(':', '-').replace('.zip', '')
            wav = os.path.join(args.save_root, args.lan_pair, args.split, f"{uttid}.wav")

        # transcribe with Whisper 
        result = model.transcribe(wav)
        
        # write to output 
        target_f = os.path.join(args.save_root, args.lan_pair, args.split, f"{uttid}.txt")
        with open(target_f, 'w') as f: 
            f.write('%s\n' % result['text'])
        
