import json
import concurrent.futures
import requests

with open('config.json') as fp:
    config = json.load(fp)
max_length = config['max_length']
batch_size = config['batch_size']
name = f'bert-max_length{max_length}-batch_size{batch_size}'

# dispatch requests in parallel
url = f'http://localhost:8080/predictions/{name}'
paraphrase = {'seq_0': "HuggingFace's headquarters are situated in Manhattan",
        'seq_1': "The company HuggingFace is based in New York City"}
not_paraphrase = {'seq_0': paraphrase['seq_0'], 'seq_1': 'This is total nonsense.'}

with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    def worker_thread(worker_index):
        # we'll send half the requests as not_paraphrase examples for sanity
        data = paraphrase if worker_index < batch_size//2 else not_paraphrase
        try:
            response = requests.post(url, data=data)

            # Check if the response status code indicates success
            if response.status_code == 200:
                print(worker_index, response.json())
            else:
                # If the response is not successful, raise an exception with the status code and error message
                error_message = response.json().get('message', 'Unknown Error')
                raise Exception(f"Failed request with status code {response.status_code}: {error_message}")
        except Exception as e:
            # Catch all other exceptions that may be raised
            print(f"An unexpected error occurred: {e}")
            raise

    for worker_index in range(batch_size):
        executor.submit(worker_thread, worker_index)
