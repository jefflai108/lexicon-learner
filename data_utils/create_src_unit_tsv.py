import pandas as pd

def fix_train_mined_tsv(target_tsv, source_tsv, output_target_tsv):

    src_df = pd.read_table(source_tsv, sep='\t', header=0)
    tgt_df = pd.read_table(target_tsv, sep='\t', header=0)

    out_tgt_f = open(output_target_tsv, 'w')

    out_tgt_f.write('id\tscore\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n')

    num_examples = len(src_df['tgt_text'])

    for i in range(num_examples): 
        assert len(tgt_df['tgt_audio'][i].split()) == tgt_df['tgt_n_frames'][i]
        src_token_seq = src_df['tgt_text'][i]
        src_token_len = len(src_token_seq.split())
        out_tgt_f.write('%s\t%s\t%s\t%d\t%s\t%s\n' % (tgt_df['id'][i], tgt_df['score'][i], src_token_seq, src_token_len, tgt_df['tgt_audio'][i], tgt_df['tgt_n_frames'][i]))

    out_tgt_f.close()


def fix_valid_vp_tsv(target_tsv, source_tsv, output_target_tsv): 

    src_df = pd.read_table(source_tsv, sep='\t', header=0)
    tgt_df = pd.read_table(target_tsv, sep='\t', header=0)

    out_tgt_f = open(output_target_tsv, 'w')

    out_tgt_f.write('id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n')
    
    num_examples = len(src_df['tgt_text'])

    for i in range(num_examples): 
        if isinstance(tgt_df['tgt_audio'][i], str):
            assert len(tgt_df['tgt_audio'][i].split()) == tgt_df['tgt_n_frames'][i]
        src_token_seq = src_df['tgt_text'][i]
        src_token_len = len(src_token_seq.split())
        out_tgt_f.write('%s\t%s\t%d\t%s\t%s\n' % (tgt_df['id'][i], src_token_seq, src_token_len, tgt_df['tgt_audio'][i], tgt_df['tgt_n_frames'][i]))

    out_tgt_f.close()


if __name__ == '__main__': 
    # filter fairseq manifest files based on target utterance # of frames 

    import argparse 
    parser = argparse.ArgumentParser('Filter fairseq manifest')

    parser.add_argument('--lan_pair', type=str, default='es-en')
    parser.add_argument('--data_filter_threshold', type=float, default=1.09)
    parser.add_argument('--frame_threshold', type=int, default=1024)
    parser.add_argument('--data_root', type=str, default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/s2u_manifests')
    args = parser.parse_args()
   
    # e.g.
    #lan_pair = 'es-en'
    #data_filter_threshold = 1.09

    #lan_pair = 'ro-en'
    #data_filter_threshold = 1.07
    
    fix_train_mined_tsv(f'{args.data_root}/{args.lan_pair}/train_mined_t{args.data_filter_threshold}_filter{args.frame_threshold}.tsv', 
                        f'{args.data_root}/{args.lan_pair}/source_unit/train_mined_t{args.data_filter_threshold}_filter{args.frame_threshold}.tsv', 
                        f'{args.data_root}/{args.lan_pair}/train_mined_t{args.data_filter_threshold}_filter{args.frame_threshold}_u2u.tsv') 
                    
    fix_valid_vp_tsv(f'{args.data_root}/{args.lan_pair}/valid_vp_filter{args.frame_threshold}.tsv', 
                     f'{args.data_root}/{args.lan_pair}/source_unit/valid_vp_filter{args.frame_threshold}.tsv', 
                     f'{args.data_root}/{args.lan_pair}/valid_vp_filter{args.frame_threshold}_u2u.tsv')

    fix_valid_vp_tsv(f'{args.data_root}/{args.lan_pair}/test_epst_filter{args.frame_threshold}.tsv',
                     f'{args.data_root}/{args.lan_pair}/source_unit/test_epst_filter{args.frame_threshold}.tsv',
                     f'{args.data_root}/{args.lan_pair}/test_epst_filter{args.frame_threshold}_u2u.tsv')

    fix_valid_vp_tsv(f'{args.data_root}/{args.lan_pair}/test_fleurs.tsv',
                     f'{args.data_root}/{args.lan_pair}/source_unit/test_fleurs.tsv',
                     f'{args.data_root}/{args.lan_pair}/test_fleurs_u2u.tsv')

    fix_valid_vp_tsv(f'{args.data_root}/{args.lan_pair}/test_epst.tsv',
                     f'{args.data_root}/{args.lan_pair}/source_unit/test_epst.tsv',
                     f'{args.data_root}/{args.lan_pair}/test_epst_u2u.tsv')
