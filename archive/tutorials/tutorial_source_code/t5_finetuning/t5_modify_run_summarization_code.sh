#!/bin/bash
set -eExuo pipefail

cd ~/transformers/examples/pytorch/summarization

# Insert code into run summarization to disable DDP for torchrun
tee temp_run_summarization.py > /dev/null <<EOF
# Disable DDP for torchrun
from transformers import __version__, Trainer
Trainer._wrap_model = lambda self, model, training=True, dataloader=None: model
EOF

cat run_summarization.py >> temp_run_summarization.py
mv temp_run_summarization.py run_summarization.py
chmod +x run_summarization.py