from typing import Optional

INSTANCETYPE_TO_NEURONCORES = {
    "inf1.xlarge": 4,
    "inf1.2xlarge": 4,
    "inf1.6xlarge": 16,
    "inf2.xlarge": 2,
    "inf2.8xlarge": 2,
    "inf2.24xlarge": 12,
    "inf2.48xlarge": 24,
    "inf1.24xlarge": 64,
    "trn1.2xlarge": 2,
    "trn1.32xlarge": 32,
}

def get_instance_type() -> str:
    """Try to obtain the instance type."""
    try:
        from urllib.request import Request, urlopen

        req = Request("http://169.254.169.254/latest/api/token", method="PUT")
        req.add_header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
        with urlopen(req) as response:
            token = response.read().decode("utf-8")

        req = Request("http://169.254.169.254/latest/meta-data/instance-type")
        req.add_header("X-aws-ec2-metadata-token", token)
        with urlopen(req) as response:
            instance_type = response.read().decode("utf-8")

        return instance_type
    except:  # noqa: E722, there are various ways above code can fail and we don't care
        return None

def get_num_neuroncores(instance_type: Optional[str] = None) -> int:
    """
    Try to obtain the maximum number of NeuronCores available on this instance.

    Args:
        instance_type: The Neuron instance type. Autodetermined from current instance
            if not provided.

    Returns:
        The number of NeuronCores (or 2 if the type is unknown).
    """

    try:
        if not instance_type:
            instance_type = get_instance_type()
        return INSTANCETYPE_TO_NEURONCORES[instance_type]
    except KeyError:
        num_cores = 2
        return num_cores