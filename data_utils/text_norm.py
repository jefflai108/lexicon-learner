from num2words import num2words
import re, glob, os, string 

def normalize(sentence): 
    # Remove punctuation from the sentence
    clean_sentence = ''.join([char for char in sentence if char not in string.punctuation])

    # Convert numbers to spoken form
    ordinal_regex = r'\d+(st|nd|rd|th)'
    number_regex = r'\d+(?:\.\d+)?'

    clean_sentence = re.sub(ordinal_regex, lambda match: num2words(int(match.group(0)[:-2]), ordinal=True), clean_sentence)
    clean_sentence = re.sub(number_regex, lambda match: num2words(float(match.group(0))), clean_sentence)

    return clean_sentence

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser('Apply text-norm to existing transcriptions')

    parser.add_argument('--split', type=str, default='en-test_epst')
    parser.add_argument('--lan_pair', type=str, default='s2u_en-es')
    parser.add_argument('--save_root', type=str, default='/data/sls/temp/clai24/data/speech_matrix/speech_to_unit/mfa_s2u_manifests/')
    args = parser.parse_args()


    # Iterate through all the text files in the directory
    for file_path in glob.glob(os.path.join(args.save_root, args.lan_pair, args.split, "*.txt")):
        ## Open the file in read mode
        with open(file_path, "r") as file:
            # Read the original content and modify each line
            lines = file.readlines()
            norm_lines = normalize(lines[0])
            print(norm_lines)
            print("%s\n" % norm_lines)


        ## Write the modified content back to the file
        with open(file_path, "w") as file:
            file.write("%s\n" % norm_lines)

