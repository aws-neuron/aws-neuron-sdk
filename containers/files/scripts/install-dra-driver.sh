#!/bin/bash

# Deploy Neuron DRA Driver
set -e

echo "🚀 Deploying Neuron DRA Driver..."

# Check argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <image_name>"
    echo "Example: $0 123456789.dkr.ecr.us-west-2.amazonaws.com/neuron-dra-driver:v1.0"
    exit 1
fi

# Get the script directory and set the manifests path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="$SCRIPT_DIR/../../manifests"
DRA_IMAGE="$1"

# Apply all manifests in order
echo "📝 Creating namespace..."
kubectl apply -f "$MANIFESTS_DIR/namespace.yaml"

echo "🔐 Creating ServiceAccount and RBAC..."
kubectl apply -f "$MANIFESTS_DIR/serviceaccount.yaml"
kubectl apply -f "$MANIFESTS_DIR/clusterrole.yaml"
kubectl apply -f "$MANIFESTS_DIR/clusterrolebinding.yaml"

echo "📱 Creating DeviceClass..."
kubectl apply -f "$MANIFESTS_DIR/deviceclass.yaml"

echo "🔧 Deploying DRA DaemonSet..."
# Check if DaemonSet already exists before applying
DAEMONSET_EXISTS=false
if kubectl get daemonset neuron-dra-driver-kubelet-plugin -n neuron-dra-driver >/dev/null 2>&1; then
    DAEMONSET_EXISTS=true
    echo "📋 DaemonSet already exists, will restart after applying..."
fi

echo "🏷️  Using custom image: $DRA_IMAGE"
sed "s|NEURON_DRA_IMAGE|$DRA_IMAGE|g" "$MANIFESTS_DIR/daemonset.yaml" | kubectl apply -f -

# If DaemonSet was already running, restart it to pull latest image
if [ "$DAEMONSET_EXISTS" = true ]; then
    echo "🔄 Restarting DaemonSet to pull latest image..."
    kubectl rollout restart daemonset/neuron-dra-driver-kubelet-plugin -n neuron-dra-driver
    echo "⏳ Waiting for rollout to complete..."
    kubectl rollout status daemonset/neuron-dra-driver-kubelet-plugin -n neuron-dra-driver --timeout=300s
else
    echo "⏳ Waiting until pods are in a running state..."
    kubectl wait --for=condition=ready pod -l app=neuron-dra-driver-kubelet-plugin -n neuron-dra-driver --timeout=300s
fi

echo "✅ Deployment complete!"

echo ""
echo "📊 Recent logs from dra driver:"
kubectl logs -n neuron-dra-driver -l app=neuron-dra-driver-kubelet-plugin --tail=10
echo ""