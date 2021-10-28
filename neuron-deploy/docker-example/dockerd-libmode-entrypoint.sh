#!/bin/bash
if [[ "$1" = "serve" ]]; then
  # Start your application here!
  # e.g: 'python my_server_app.py'
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
