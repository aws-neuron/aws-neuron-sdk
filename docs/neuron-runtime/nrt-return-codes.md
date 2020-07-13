# Neuron Runtime return codes

|Code	|Name |Description	|
|---	|---	|---	|
|0	|OK	|Successful completion.	|
|1	|FAIL	|Non specific failure, e.g. low level interface to the hardware error.	|
|2	|INVALID	|Invalid/corrupted NEFF file, bad Neuron instruction, invalid access to Neuron memory, etc.	|
|3	|INVALID_HANDLE	|Invalid handle was passed to the request, e.g. for a Neural Network that has not been loaded or has been previously unloaded.	|
|4	|RESOURCE	|Failed to allocate a resource for requested operation, for example: not enough TDRAM to load a Neural Network, not enough Host memory to perform an operation.	|
|5	|TIMEOUT	|Request timed out.	|
|6	|HW_ERROR	|RT failed to initialize, after a number of failed attempts to start RT stays up and returns the error in response to every GRPC call.  Common causes: bad hugepages configuration (insufficient number of reserved pages on the system), failure to initialize Neuron device.	|
|7	|QUEUE_FULL	|Not enough space in the inference input queue.  The number of submitted and not completed inference requests is greater that what’s been configured during NN load.  This is a transient error, inference requests can be submitted after some of the in-flight inferences have completed.	|
|9	|RESOURCE_NC	|Not enough available NCs to load a Neural Network.	|
|10	|UNSUPPORTED_VERSION	|NN load failed because the version of NEFF is not supported.	|
|1000	|INFER_PENDING	|Inference has not completed yet.	|
|1002	|INFER_BAD_INPUT	|Invalid input has been submitted to infer, e.g. missing some of the input tensors, incorrect input tensor sizes.	|
|1003	|INFER_COMPLETED_WITH_NUM_ERR	|Inference is completed.  Numerical errors were encountered while executing the inference (NaN).	|
|1004	|INFER_COMPLETED_WITH_ERR	|Inference is completed.  Non-numerical errors were encoutered while executing the inference.  Usually indicates problems with compiled NN (compiler errors) or the hardware issues.	|

