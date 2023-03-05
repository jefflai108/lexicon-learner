import sys 
import json 
import torch 
import torch.nn.functional as F
import numpy as np 

def main(lex_align_file, lex_align_prob_file, src_vocab_size=1000, tgt_vocab_size=1000, fairseq_dict_offset=4, softmax_temp=0.1):

    # Fairseq dictionary prepended these symbols (https://github.com/facebookresearch/fairseq/blob/0338cdc3094ca7d29ff4d36d64791f7b4e4b5e6e/fairseq/data/dictionary.py#L34-L37)
    BOS_TOKEN = 0 
    PAD_TOLEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3

    lex_align_cnt = read_json_file(lex_align_file)
    lex_align_prob = np.zeros((src_vocab_size, tgt_vocab_size+fairseq_dict_offset))

    # find unaligned src tokens, map them to Fairseq <unk>
    unaligned_src_tokens = [] 
    for key in range(src_vocab_size): 
        key = str(key)
        if key not in lex_align_cnt.keys(): 
            unaligned_src_tokens.append(key) 

    for token in unaligned_src_tokens:
        lex_align_prob[int(token)][UNK_TOKEN] = 1.0

    # convert aligned token_cnt into probabilities 
    for (src,tgt_dict_cnt) in lex_align_cnt.items(): 
        src = int(src)
        tgt_aligned_tokens = np.array([int(j) + fairseq_dict_offset for j in tgt_dict_cnt.keys()])
        tgt_aligned_token_cnt = torch.tensor(np.array(list(tgt_dict_cnt.values())), dtype=torch.float)
        lex_align_prob[src][tgt_aligned_tokens] = F.softmax(tgt_aligned_token_cnt / softmax_temp, dim=0).numpy()

    # print some statistics 
    copy_translation = []
    for i in range(src_vocab_size): 
        tgt_idx = np.argmax(lex_align_prob[i]) - fairseq_dict_offset
        if i == tgt_idx: 
            copy_translation.append(i)
    print('there are %d src tokens that mapped directly to its corresponding tgt token (copy translation)' % len(copy_translation))

    # store the resulting np array 
    np.save(lex_align_prob_file, lex_align_prob)

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
