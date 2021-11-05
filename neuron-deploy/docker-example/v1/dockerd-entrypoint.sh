#!/bin/bash
set -e

wait_for_nrtd() {
  nrtd_sock="/run/neuron.sock"
  SOCKET_TIMEOUT=300
  is_wait=true
  wait_time=0
  i=1
  sp="/-\|"
  echo -n "Waiting for neuron-rtd  "
  pid=$1
  while $is_wait; do
    if [ -S "$nrtd_sock" ]; then
      echo "$nrtd_sock Exist..."
      is_wait=false
    else
      sleep 1
      wait_time=$((wait_time + 1))
      if [ "$wait_time" -gt "$SOCKET_TIMEOUT" ]; then
        echo "neuron-rtd failed to start, exiting"
	      cat /tmp/nrtd.log
        exit 1
      fi
      printf "\b${sp:i++%${#sp}:1}"
    fi
  done
  cat /tmp/nrtd.log
}

# Start neuron-rtd
/opt/aws/neuron/bin/neuron-rtd -g unix:/run/neuron.sock --log-console  >>  /tmp/nrtd.log 2>&1 &
nrtd_pid=$!
echo "NRTD PID: "$nrtd_pid""
#wait for nrtd to be up (5 minutes timeout)
wait_for_nrtd $nrtd_pid
export NEURON_RTD_ADDRESS=unix:/run/neuron.sock
nrtd_present=1

if [[ "$1" = "serve" ]]; then
  # Start your application here!
  # e.g: 'python my_server_app.py'
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
