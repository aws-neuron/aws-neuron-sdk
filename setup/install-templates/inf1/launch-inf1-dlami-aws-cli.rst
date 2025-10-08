.. _launch-inf1-dlami-aws-cli:

AWS CLI commands to launch inf1 instances
"""""""""""""""""""""""""""""""""""""""""

.. code:: bash

  # Launch instance
  # The following are the different Deep Learning AMIs to get started and is recommended
  # for the tutorials.
  # "Deep Learning AMI (Amazon Linux)*"
  # "Deep Learning AMI (Amazon Linux 2)*"
  # "Deep Learning AMI (Ubuntu 18.04)*"
  #

  # You can get the latest AMI ID for any of the above ones using the following command
  AWS_REGION="<aws region name like us-east-1>"
  AMIID=$(aws ec2 describe-images  --filters "Name=name,Values=Deep Learning Base AMI (Ubuntu 18.04)*" --query 'sort_by(Images, &CreationDate)[].[Name,ImageId]' --region $AWS_REGION --output text | tail -n 1  | awk '{print $(NF)}')

  INSTANCE_ID=$(aws ec2 run-instances --image-id $AMIID --count 1 --instance-type <inf1.xlarge type> --key-name MyKeyPair --region $AWS_REGION [--subnet-id <subnet id>]| python -c 'import sys, json; print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')
  echo "Instance ID of launched instance" $INSTANCE_ID

  # Wait for few seconds to a minute for the instance to get created and have public DNS/ip.

  # The following command will get the public DNS name of the launched instance to which
  # you can then log in to using your key pair.
  INSTANCE_PUBLIC_DNS=$(aws ec2 describe-instances --instance-id $INSTANCE_ID --region $AWS_REGION | python -c 'import sys, json; print(json.load(sys.stdin)["Reservations"][0]["Instances"][0]["PublicDnsName"])')
  echo "DNS name of the launched instance" $INSTANCE_PUBLIC_DNS

  # Wait for couple of minutes for the instance to be ready and then login:
  ssh -i <key.pem> <ubuntu/ec2-user>@$INSTANCE_PUBLIC_DNS


