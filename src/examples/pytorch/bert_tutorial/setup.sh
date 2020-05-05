#! /bin/bash
if [[ "$#" -ne 1 ]]; then 
    echo "Usage: setup.sh <s3_bucket_name_to_use>"
    echo 
    echo "This will download the required materials into the current working"
    echo "directory in your conda environment, and create (or confirm) your"
    echo "S3 bucket exists"
    return 1
fi

echo "Get AWS account ID to postfix bucker name"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

if [ "$ACCOUNT" == "" ]; then
    echo "Failed to fetch account ID for unique bucket name"
    return 1
else
    echo "Fetched account ID = $ACCOUNT"
fi

if ! [[ -x "$(command -v git)" ]]; then
    echo "The aws cli needs to be installed.  Aborting."
    return 1
fi

if ! [[ -f "$HOME/.aws/credentials" ]]; then
    echo "AWS credentials file not found, please use 'aws configure'"
    return 1
fi

BUCKET_NAME="$1-$ACCOUNT"

echo 
echo "This script will set your environment and create an upload S3 bucket called $BUCKET_NAME, continue (yes/no)?"
read CONTINUE

if [[ ! -z "$CONTINUE" ]] && [[ "$CONTINUE" == "yes" ]]; then
    echo "Installing ..."
    echo
else
    echo "Exiting ..."
    echo
    return 1
fi

## Activate DLAMI conda environment
echo "Activate CONDA Environment"
source activate pytorch_p36

# Get GLUE data
echo "Get GLUE data ..."
$(wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py >/dev/null 2>&1)

if [[ -f "download_glue_data.py" ]]; then 
    if [[ $(python download_glue_data.py) ]]; then
        echo "Successfully setup GLUE data"
    else
        "Failed to setup GLUE data"
        return 1
    fi
else
    echo "Failed to fetch GLUE download script"
    return 1
fi

# Get a local copy of the transformers code
echo "Get Transformers ..."
if [ -d "transformers" ]; then
    echo "Directory 'transformers' already exists, not downloading from github"
else
    git clone https://github.com/huggingface/transformers
    if [ -d "transformers" ]; then
        echo "GIT clone of transformers succeeded"
    else
        echo "GIT clone of tranformers failed"
        return 1
    fi
fi

echo "Install Transformers ..."
if [[ $(pip install -e transformers) ]]; then
    echo "Successfully installed local transformers in conda environment"
else
    echo "Failed to install transformers in local environment"
    return 1
fi

# Install scikit learn from CONDA (this is easier than making pip install work)
echo "Install Scikit Learn ..."
if [[ $(conda install scikit-learn -y) ]]; then
    echo "Successfully install scikit learn"
else
    echo "Failed to install scikit learn"
    return 1
fi

echo "Install PyTorch 1.3.1 ..."
if [[ $(pip install torch==1.3.1) ]]; then
    echo "Successfully installed torch 1.3.1"
else
    echo "Failed to install torch 1.3.1"
    return 1
fi

echo "Install transformer requirements (slow ~10 mins includes build steps) ..."
if [[ $(pip install -r transformers/examples/requirements.txt) ]]; then
    echo "Successfully installed requirements.txt"
else
    echo "Failed to install requirements.txt"
    return 1
fi

echo "Create S3 Bucket $BUCKET_NAME ..."
if output="$(aws s3 ls $BUCKET_NAME)"; then
    echo "Bucket $1 already exists"
else
    echo "Bucket did not exist. Creating bucket $BUCKET_NAME"
    if output=$(aws s3 mb s3://$BUCKET_NAME); then
        echo "Successfully created bucket $BUCKET_NAME"
    else
        echo "Failed to create bucket $BUCKET_NAME"
        echo output
        return 1
    fi
fi

echo "s3://$BUCKET_NAME" > bert_bucket_name.txt
