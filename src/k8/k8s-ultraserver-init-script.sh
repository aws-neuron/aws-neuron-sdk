#!/bin/bash

MPI_HOST_FILE=/etc/mpi/hostfile

NEURON_ULTRASERVER_MODE_UNSET=0
NEURON_ULTRASERVER_MODE_X4=1
NEURON_ULTRASERVER_MODE_X2H=2
NEURON_ULTRASERVER_MODE_X2V=3
NEURON_ULTRASERVER_MODE_X1=4

ULTRASERVER_INIT_DIR=/root/ultraserver_init
SORTED_NODES_FILE=$ULTRASERVER_INIT_DIR/sorted_nodes.txt
FQDN_MODE_FILE=$ULTRASERVER_INIT_DIR/fqdn_mode.txt
ENV_VARS_FILE=$ULTRASERVER_INIT_DIR/us_env_vars.txt
NEW_HOST_FILE=$ULTRASERVER_INIT_DIR/new_hostfile

export NEURON_ULTRASERVER_SERVER_ID_DEFAULT_VALUE="0000000000000000"
export NEURON_ULTRASERVER_NODE_ID_DEFAULT_VALUE=-1

export NEURON_GLOBAL_TOPOID0_HOST=""

export NUM_WORKERS=0

cat /dev/null > $SORTED_NODES_FILE
cat /dev/null > $FQDN_MODE_FILE
cat /dev/null > $ENV_VARS_FILE
cat /dev/null > $NEW_HOST_FILE

save_sorted_node_list() {
    # Gather ultraserver information from each worker node
    mpirun --allow-run-as-root \
        --mca orte_keep_fqdn_hostnames 1 \
        -host $ip_list \
        -x NEURON_ULTRASERVER_SERVER_ID_DEFAULT_VALUE \
        -x NEURON_ULTRASERVER_NODE_ID_DEFAULT_VALUE \
        -x NEURON_ULTRASERVER_NODE_CONFIG \
        sh -c '
            if [ -f "/sys/class/neuron_device/server_id_${NEURON_ULTRASERVER_NODE_CONFIG}" ]; then
                NEURON_ULTRASERVER_SERVER_ID=$(cat /sys/class/neuron_device/server_id_${NEURON_ULTRASERVER_NODE_CONFIG})
            else
                NEURON_ULTRASERVER_SERVER_ID=$NEURON_ULTRASERVER_SERVER_ID_DEFAULT_VALUE
            fi

            if [ -f "/sys/class/neuron_device/node_id_${NEURON_ULTRASERVER_NODE_CONFIG}" ]; then
                NEURON_ULTRASERVER_NODE_ID=$(cat /sys/class/neuron_device/node_id_${NEURON_ULTRASERVER_NODE_CONFIG})
            else
                NEURON_ULTRASERVER_NODE_ID=$NEURON_ULTRASERVER_NODE_ID_DEFAULT_VALUE
            fi

            FQDN=$(hostname --fqdn)
            echo $NEURON_ULTRASERVER_SERVER_ID:$NEURON_ULTRASERVER_NODE_ID:$FQDN
        ' | sort -t':' -k1,1 -k2,2 -k3,3 > $SORTED_NODES_FILE

    # Set the topology ids for each worker node
    local i=0
    while IFS= read -r line; do
        echo "${i}:${line}"
        ((i++))
    done < $SORTED_NODES_FILE > temp && mv temp $SORTED_NODES_FILE
    NEURON_GLOBAL_TOPOID0_HOST=$(head -n1 $SORTED_NODES_FILE | cut -d: -f4)
}

validate_node_config() {
    while read -r server_id; do
        # Server id and node id are only valid for node configs > 1
        if [ $NEURON_ULTRASERVER_NODE_CONFIG -ne 1 ]; then
            # Validate server id exists
            if [ "$server_id" = "$NEURON_ULTRASERVER_SERVER_ID_DEFAULT_VALUE" ]; then
                echo "$NEURON_ULTRASERVER_NODE_CONFIG-node config is not supported"
                exit 1
            fi

            # Validate there is the correct amount of nodes that share the same server id
            count=$(grep "$server_id" "$SORTED_NODES_FILE" | wc -l)
            if [ $count -ne $NEURON_ULTRASERVER_NODE_CONFIG ]; then
                echo "Error: Incorrect number of nodes with server id $server_id, need $NEURON_ULTRASERVER_NODE_CONFIG nodes but saw $count"
                exit 1
            fi

            # Validate all the node ids are unique
            node_ids_count=$(grep "$server_id" "$SORTED_NODES_FILE" | cut -d':' -f3 | sort | uniq | wc -l)
            if [ $node_ids_count -ne $NEURON_ULTRASERVER_NODE_CONFIG ]; then
                echo "Error: Found $node_ids_count unique node IDs, expected $NEURON_ULTRASERVER_NODE_CONFIG"
                exit 1
            fi
        fi

        while IFS=':' read -r tid sid nid fqdn; do
            # Validate mode is valid for each node
            modes="${fqdn_modes_map[$fqdn]}"
            if [ $NEURON_ULTRASERVER_NODE_CONFIG -eq 4 ]; then
                if echo "$modes" | grep -q "\b$NEURON_ULTRASERVER_MODE_X4\b"; then
                    mode=$NEURON_ULTRASERVER_MODE_X4
                else
                    echo "Error: Node $fqdn does not support 4-node config"
                    exit 1
                fi
            elif [ $NEURON_ULTRASERVER_NODE_CONFIG -eq 2 ]; then
                if echo "$modes" | grep -q "\b$NEURON_ULTRASERVER_MODE_X2V\b"; then
                    mode=$NEURON_ULTRASERVER_MODE_X2V
                elif echo "$modes" | grep -q "\b$NEURON_ULTRASERVER_MODE_X2H\b"; then
                    mode=$NEURON_ULTRASERVER_MODE_X2H
                else
                    echo "Error: Node $fqdn does not support 2-node config"
                    exit 1
                fi
            else
                mode=$NEURON_ULTRASERVER_MODE_X1
            fi

            # Save each worker node's environments variables to a file
            echo "${tid}:${mode}:${sid}:${nid}:${fqdn}" >> "$ENV_VARS_FILE"
        done < <(grep "$server_id" "$SORTED_NODES_FILE")
    done < <(cut -d':' -f2 "$SORTED_NODES_FILE" | sort | uniq)
}

reorder_hostfile() {
    # Check if files exist
    if [ ! -f "$MPI_HOST_FILE" ] || [ ! -f "$SORTED_NODES_FILE" ]; then
        echo "Error: One or both input files do not exist"
        exit 1
    fi

    # Extract FQDNs from SORTED_NODES_FILE and reorder entries
    while IFS=: read -r _ _ _ fqdn; do
        # Remove .cluster.local suffix
        clean_fqdn=${fqdn%.cluster.local}

        # Find the matching line in original file
        while read -r line; do
            if [[ "$line" == "$clean_fqdn"* ]]; then
                echo "$line" >> "$NEW_HOST_FILE"
                break
            fi
        done < "$MPI_HOST_FILE"
    done < "$SORTED_NODES_FILE"
}

# Validate node config
if [ -z "${NEURON_ULTRASERVER_NODE_CONFIG}" ]; then
    NEURON_ULTRASERVER_NODE_CONFIG=4
fi
if [ $NEURON_ULTRASERVER_NODE_CONFIG -ne 1 ] && [ $NEURON_ULTRASERVER_NODE_CONFIG -ne 2 ] && [ $NEURON_ULTRASERVER_NODE_CONFIG -ne 4 ]; then
    echo "Error: Invalid ultraserver node config: $NEURON_ULTRASERVER_NODE_CONFIG. Must be 1, 2, or 4."
    exit 1
fi
echo "Using $NEURON_ULTRASERVER_NODE_CONFIG-node config"

echo -e "\nCurrent hostfile:"
cat $MPI_HOST_FILE

# Read the file, extract the first column, resolve IPs, and build the comma-separated string
ip_list=""
while read line; do
    ip=$(getent hosts "$line" | awk '{print $1}')
    if [ -z "$ip" ]; then
        echo "error: Unable to resolve IP address for host: $line"
        exit 1
    fi
    if [ -z "$ip_list" ]; then
        ip_list="$ip"
    else
        ip_list="${ip_list},${ip}"
    fi
done < <(cut -d' ' -f1 $MPI_HOST_FILE)
echo "Worker pod IPs:" "$ip_list"

# Count unique IPs from ip_list and store in NUM_WORKERS
NUM_WORKERS=$(echo "$ip_list" | tr -cd ',' | wc -c)
NUM_WORKERS=$((NUM_WORKERS + 1))
echo "Number of worker nodes: $NUM_WORKERS"

# Validate that the number of workers is a multiple of the node config
if [ $((NUM_WORKERS % NEURON_ULTRASERVER_NODE_CONFIG)) -ne 0 ]; then
    echo "Error: Invalid number of worker nodes for $NEURON_ULTRASERVER_NODE_CONFIG-node config: $NUM_WORKERS."
    exit 1
fi

# Create a map of workers to their possible ultraserver modes
mpirun --allow-run-as-root \
    --mca orte_keep_fqdn_hostnames 1 \
    -host $ip_list \
    sh -c '
        FQDN=$(hostname --fqdn)
        NEURON_ULTRASERVER_MODE=$(cat /sys/class/neuron_device/ultraserver_mode)
        echo $FQDN:$NEURON_ULTRASERVER_MODE
    ' | sort -t':' -k1 > $FQDN_MODE_FILE
declare -A fqdn_modes_map
while IFS=':' read -r fqdn mode; do
    fqdn_modes_map["$fqdn"]="$mode"
done < $FQDN_MODE_FILE
(echo "FQDN:Modes" && cat $FQDN_MODE_FILE) | tr ':' '    '

# Validate worker nodes
echo -e "\nSorted nodes:"
save_sorted_node_list
(echo "TOPO_ID:SERVER_ID:NODE_ID:FQDN" && cat $SORTED_NODES_FILE) |  tr ':' '    '
echo -e "\nNEURON_GLOBAL_TOPOID0 node will be: $NEURON_GLOBAL_TOPOID0_HOST"
validate_node_config

# Update hostlist
echo -e "\nUpdated hostfile:"
reorder_hostfile
cat $NEW_HOST_FILE

# Write environment variables to each worker node
for line in `cat $ENV_VARS_FILE`; do
    IFS=':' read -r topo_id mode server_id node_id fqdn <<< "$line"
    export mode server_id node_id fqdn topo_id
    mpirun --allow-run-as-root \
        --mca orte_keep_fqdn_hostnames 1 \
        -host $fqdn \
        -x topo_id \
        -x NEURON_GLOBAL_TOPOID0_HOST \
        -x mode \
        -x server_id \
        -x node_id \
        sh -c '
            sed -i "/^NEURON_GLOBAL_TOPOID=/d" /etc/environment
            sed -i "/^NEURON_GLOBAL_TOPOID0_HOST=/d" /etc/environment
            sed -i "/^NEURON_RT_ULTRASERVER_MODE=/d" /etc/environment
            sed -i "/^NEURON_RT_ULTRASERVER_SERVER_ID=/d" /etc/environment
            sed -i "/^NEURON_RT_ULTRASERVER_NODE_ID=/d" /etc/environment

            echo "NEURON_GLOBAL_TOPOID=$topo_id" >> /etc/environment
            echo "NEURON_GLOBAL_TOPOID0_HOST=$NEURON_GLOBAL_TOPOID0_HOST" >> /etc/environment
            echo "NEURON_RT_ULTRASERVER_MODE=$mode" >> /etc/environment
            echo "NEURON_RT_ULTRASERVER_SERVER_ID=$server_id" >> /etc/environment
            echo "NEURON_RT_ULTRASERVER_NODE_ID=$node_id" >> /etc/environment

            echo "Node $(hostname --fqdn): Variables set and persisted"
            echo "NEURON_GLOBAL_TOPOID=$topo_id"
            echo "NEURON_GLOBAL_TOPOID0_HOST=$NEURON_GLOBAL_TOPOID0_HOST"
            echo "NEURON_RT_ULTRASERVER_MODE=$mode"
            echo "NEURON_RT_ULTRASERVER_SERVER_ID=$server_id"
            echo "NEURON_RT_ULTRASERVER_NODE_ID=$node_id"
        '
done
