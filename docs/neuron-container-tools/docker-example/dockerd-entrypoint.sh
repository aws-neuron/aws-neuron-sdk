#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    multi-model-server --start --model-store /home/model-server/tmp/models
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
