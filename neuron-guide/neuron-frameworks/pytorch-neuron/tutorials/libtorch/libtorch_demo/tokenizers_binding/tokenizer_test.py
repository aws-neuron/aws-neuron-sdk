from transformers import AutoTokenizer
import argparse
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_tests', type=int, default=10_000)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')

start = time.time()
for _ in tqdm(range(args.num_tests), desc='Tokenizing'):
    tokenizer.encode("If everything goes smoothly, this text will be tokenized inside Rust.")
end = time.time()
print('Python took {:.2f} seconds.'.format(end - start))
