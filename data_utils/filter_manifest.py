import pandas as pd

def read_train_tsv(target_tsv, source_tsv, output_target_tsv, output_source_tsv, tgt_frame_filter_threshold=100): 

    src_df = pd.read_table(source_tsv, sep='\t', header=0)
    tgt_df = pd.read_table(target_tsv, sep='\t', header=0)

    out_tgt_f = open(output_target_tsv, 'w')
    out_src_f = open(output_source_tsv, 'w')

    out_tgt_f.write('id\tscore\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n')
    out_src_f.write('id\ttgt_text\tscore\n')

    num_examples = len(src_df['tgt_text'])

    for i in range(num_examples): 
        tgt_n_frames = tgt_df['tgt_n_frames'][i]

        if tgt_n_frames <= tgt_frame_filter_threshold: 
            out_tgt_f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (tgt_df['id'][i], tgt_df['score'][i], tgt_df['src_audio'][i], tgt_df['src_n_frames'][i], tgt_df['tgt_audio'][i], tgt_df['tgt_n_frames'][i]))
            out_src_f.write('%s\t%s\t%s\n' % (src_df['id'][i], src_df['tgt_text'][i], src_df['score'][i]))

    out_tgt_f.close()
    out_src_f.close()


def read_valid_tsv(target_tsv, source_tsv, output_target_tsv, output_source_tsv, tgt_frame_filter_threshold=100): 

    src_df = pd.read_table(source_tsv, sep='\t', header=0)
    tgt_df = pd.read_table(target_tsv, sep='\t', header=0)

    out_tgt_f = open(output_target_tsv, 'w')
    out_src_f = open(output_source_tsv, 'w')

    out_tgt_f.write('id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n')
    out_src_f.write('id\ttgt_text\n')

    num_examples = len(src_df['tgt_text'])

    for i in range(num_examples): 
        tgt_n_frames = tgt_df['tgt_n_frames'][i]

        if tgt_n_frames <= tgt_frame_filter_threshold: 
            out_tgt_f.write('%s\t%s\t%s\t%s\t%s\n' % (tgt_df['id'][i], tgt_df['src_audio'][i], tgt_df['src_n_frames'][i], tgt_df['tgt_audio'][i], tgt_df['tgt_n_frames'][i]))
            out_src_f.write('%s\t%s\n' % (src_df['id'][i], src_df['tgt_text'][i]))

    out_tgt_f.close()
    out_src_f.close()

if __name__ == '__main__': 
    # filter fairseq manifest files based on target utterance # of frames 

    import argparse 
    parser = argparse.ArgumentParser('Filter fairseq manifest')

    parser.add_argument('--lan_pair', type=str, default='es-en')
    parser.add_argument('--data_filter_threshold', type=float, default=1.09)
    parser.add_argument('--frame_threshold', type=int, default=120)
    parser.add_argument('--data_root', type=str, default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests')
    args = parser.parse_args()
   
    # e.g.
    #lan_pair = 'es-en'
    #data_filter_threshold = 1.09

    #lan_pair = 'ro-en'
    #data_filter_threshold = 1.07

    read_train_tsv(f'{args.data_root}/{args.lan_pair}/train_mined_t{args.data_filter_threshold}.tsv', 
                    f'{args.data_root}/{args.lan_pair}/source_unit/train_mined_t{args.data_filter_threshold}.tsv', 
                    f'{args.data_root}/{args.lan_pair}/train_mined_t{args.data_filter_threshold}_filter{args.frame_threshold}.tsv', 
                    f'{args.data_root}/{args.lan_pair}/source_unit/train_mined_t{args.data_filter_threshold}_filter{args.frame_threshold}.tsv', 
                    args.frame_threshold)
    read_valid_tsv(f'{args.data_root}/{args.lan_pair}/valid_vp.tsv', 
                    f'{args.data_root}/{args.lan_pair}/source_unit/valid_vp.tsv', 
                    f'{args.data_root}/{args.lan_pair}/valid_vp_filter{args.frame_threshold}.tsv', 
                    f'{args.data_root}/{args.lan_pair}/source_unit/valid_vp_filter{args.frame_threshold}.tsv',
                    args.frame_threshold)
