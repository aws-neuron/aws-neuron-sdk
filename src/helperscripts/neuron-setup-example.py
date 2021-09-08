from neuronsetuphelper import neuron_setup_helper


nr_setup=neuron_setup_helper(manifest_file='default',neuron_version='latest')

setup_cmd = nr_setup.instructions(framework='tensorflow',action='Install',os='ubuntu',ami='non-dlami',mode='develop',framework_version='latest')
print (setup_cmd)
