import os
import sys
import argparse
import urllib.request

MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'

# This function is taken from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e.
def format_mrpc(data_dir, path_to_data, path_to_dev_tsv):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        try:
            mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
            mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
            urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
            urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
        except urllib.error.HTTPError:
            print("Error downloading MRPC")
            return
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    with open(mrpc_test_file, encoding='utf-8') as data_fh, \
            open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))

    dev_ids = []
    with open(path_to_dev_tsv, encoding='utf-8') as dev_fh:
        header = dev_fh.readline()
        for row in dev_fh:
            _, id1, id2, _, _ = row.strip().split('\t')
            dev_ids.append([id1, id2])

    with open(mrpc_train_file, encoding='utf-8') as data_fh, \
        open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh, \
        open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
                
    print("\tCompleted!")

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    parser.add_argument('--path_to_mrpc', help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    parser.add_argument('--path_to_dev_tsv', help='path to directory containing the glue_mrpc_dev.tsv', type=str, default='glue_mrpc_dev.tsv')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    format_mrpc(args.data_dir, args.path_to_mrpc, args.path_to_dev_tsv)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))