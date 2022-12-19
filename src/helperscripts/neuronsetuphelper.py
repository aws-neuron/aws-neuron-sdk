import json
import argparse
from packaging.version import Version, parse



########################################
# neuron_setup_helper
########################################

class neuron_release_info:
    def __init__(self):

        self.release_frameworks_all = {}
        self.release_frameworks_main = {}
        self.release_packages_all ={}
        self.release_package_main={}
        self.release_frameworks_list=[]
        self.release_components_list = []
        self.release_tf_package_to_model_server_package={}
        self.release_os_install_list =[]
        self.python_ver=""



# release_frameworks_all
# Desc: Dictionary - all framewors included in the release
#   example: 'pytorch-1.5.1': {'framework': 'pytorch', 'package': 'torch-neuron', 'version': '1.5.1.1.5.3.0', 'main': False, 'framework_version': '1.5.1', 'package_name': 'torch-neuron-1.5.1.1.5.3.0', 'pre_install_cmds': [], 'post_install_cmds': []}
# release_frameworks_all = {}

# release_frameworks_main
# Desc: Dictionary - the main frameworks in each rlease (single  version of the same framework)
#   example: 'mxnet': {'framework': 'mxnet-1.8.0', 'package': 'mx_neuron', 'version': '1.8.0.1.3.0.0', 'framework_version': '1.5.1', 'full_package_name': 'mx_neuron-1.8.0.1.3.0.0', 'pre_install_cmds': ['wget https://aws-mx-pypi.s3-us-west-2.amazonaws.com/1.8.0/aws_mx_cu110-1.8.0-py2.py3-none-manylinux2014_x86_64.whl', 'pip install aws_mx_cu110-1.8.0-py2.py3-none-manylinux2014_x86_64.whl'], 'post_install_cmds': []}
# release_frameworks_main = {}

# release_packages_all
# Desc: Dictionary -  all packages included in the release
#   example: 'aws-neuron-dkms-1.5.0.0': {'component': 'driver', 'package': 'aws-neuron-dkms', 'version': '1.5.0.0', 'main': True, 'pre_install_cmds': [], 'post_install_cmds': []}
# release_packages_all ={}

# release_package_main
# Desc: Dictionary - only single package from each component
#   example: 'driver': {'package': 'aws-neuron-dkms', 'version': '1.5.0.0', 'full_package_name': 'aws-neuron-dkms-1.5.0.0', 'pre_install_cmds': [], 'post_install_cmds': []}
# release_package_main={}


# list of all framewoks included in the specific neuron release
# release_frameworks_list=[]

# list of all neuron components included in the specific neuron release
# release_components_list = []

# dictionary to correlate tf version with model server version
# release_tf_package_to_model_server_package = {}


# list of all Neuron versions included in the manifest
neuron_ver_list = []      


# release_os_install_list =[]

dlami_conda_env= {}




package_formal_name= {
    "compiler":"Neuron Compiler",
    "tensorflow":"Neuron TensorFlow",
    "pytorch":"Neuron PyTorch",
    "mxnet":"Neuron MXNet",
    "runtime-server":"Neuron Runtime server",
    "libnrt":"Neuron Runtime library",
    "runtime-base":"Neuron Runtime base",
    "driver":"Neuron Driver",
    "tools":"Neuron Tools",
    "tensorboard":"Neuron TensorBoard",
    "tensorflow-model-server":"Neuron TensorFlow model server"
    }




########################################
# parse_arguments
########################################

def cli_parse_arguments():
    __name__='neuron-install-helper.py'
    parser = argparse.ArgumentParser(prog=__name__
    ,usage='\npython3 %(prog)s --list {neuron_versions,packages,components,frameworks} [--neuron-version=X.Y.Z]  [--file FILE] \n'
    +'python3 %(prog)s --install {pytorch,tensorflow,mxnet} [--neuron-version=X.Y.Z] [--framework-version=FRAMEWORK-X.Y.Z] [options]\n'
    +'python3 %(prog)s --install {driver,runtime,tools} [--neuron-version=X.Y.Z] [options]\n'
    +'python3 %(prog)s --update {pytorch,tensorflow,mxnet} [--framework-version=framework-X.Y.Z]  [options]\n'
    +'python3 %(prog)s --update {driver,runtime,tools} [options]\n'
    +'options= [--file FILE] [--ami {dlami,non-dlami}] [--os {ubuntu,amazonlinux}]\n'
    ,description='Installer helper for Neuron SDK')

    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--neuron-version",metavar='X.Y.Z')
    group.add_argument("--list",choices=['neuron_versions','packages','components','frameworks'])
    group.add_argument("--install",choices=['pytorch','tensorflow','mxnet'])
    group.add_argument("--update",choices=['pytorch','tensorflow','mxnet'])
    parser.add_argument("--mode",choices=['develop','compile','deploy'],default='develop')
    parser.add_argument("--framework-version",metavar='framework-X.Y.Z')
    parser.add_argument("--os",choices=['ubuntu','amazonlinux'],default='ubuntu',help='default=ubuntu')
    parser.add_argument("--ami",choices=['dlami','non-dlami'],default='non-dlami',help='default=non-dlami')
    parser.add_argument("--file",default='neuron-releases-manifest.json',help='default=neuron-releases-manifest.json')

    return parser.parse_args()




def enumerate_release_manifest(nr_setup, in_neuron_version):

    ########################################
    # Enumerate the Json file
    ########################################

    if nr_setup.file==None:
        nr_setup.file='neuron-releases-manifest.json'

    try:
        read_file = open(nr_setup.file, "r") 
    except:
        print(__name__,": error:","Can't open " + nr_setup.file + " ")
        exit(-1)

    neuron_releases = json.load (read_file)

    latest_neuron_version = neuron_releases["latest_release"]["inf1"]["version"]

    nr_setup.dlami_conda_env = neuron_releases["dlami_conda_env"]

    nr_setup.fal_supported_runtime = neuron_releases["fal_supported_runtime"]

    if (in_neuron_version == None) | (in_neuron_version == 'latest'):
        neuron_version=latest_neuron_version
    else:
        neuron_version = in_neuron_version



    for n_ver in neuron_releases["neuron_versions"]:
        neuron_ver_list.append(n_ver)



    for neuron_release_ver in neuron_releases["neuron_versions"]:
        m_release=neuron_releases["neuron_versions"][neuron_release_ver]["components"]
        n_info=neuron_release_info()
        n_info.python_ver=  neuron_releases["neuron_versions"][neuron_release_ver]["python_ver"][0]

        for component_name in m_release:
            if m_release[component_name]["framework"]==False:
                n_info.release_components_list.append(component_name)    
            m_packages=m_release[component_name]["packages"]
            for package_name in m_packages:
                for package_ver in m_packages[package_name]["versions"]:
                    m_package_ver=m_packages[package_name]["versions"][package_ver]

                    full_package_name=package_name+'-'+package_ver

                    n_info.release_packages_all[full_package_name]= {"component":component_name,"package":package_name,"version":package_ver,"main":m_package_ver["main_version"],"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}

                    if m_package_ver["main_version"]:
                        n_info.release_package_main[component_name]={"package":package_name,"version":package_ver,"full_package_name":full_package_name,"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}

                    if m_release[component_name]["framework"]:
                        ver_digits = package_ver.rsplit('.')
                        fw_ver=ver_digits[0]+'.'+ver_digits[1]+'.'+ver_digits[2]
                        fw_name_ver=component_name+'-'+fw_ver

                        if m_release[component_name]["framework"]:
                            n_info.release_components_list.append(fw_name_ver)
                            n_info.release_frameworks_list.append(fw_name_ver)

                        if m_package_ver["main_version"]:
                            n_info.release_frameworks_main[component_name]={"framework":fw_name_ver,"package":package_name,"version":package_ver,"framework_version":fw_ver,"package_name":full_package_name,"full_package_name":full_package_name,"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}


                        n_info.release_frameworks_all[fw_name_ver]={"framework":component_name,"package":package_name,"version":package_ver,"main":m_package_ver["main_version"],"framework_version":fw_ver,"package_name":full_package_name,"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}

        if 'driver' in n_info.release_components_list:
            n_info.release_os_install_list.append('driver')
        if 'runtime-server' in n_info.release_components_list:
            n_info.release_os_install_list.append('runtime-server')
        if 'tools' in n_info.release_components_list:
            n_info.release_os_install_list.append('tools')
        if 'tensorflow-model-server' in n_info.release_components_list:
            n_info.release_os_install_list.append('tensorflow-model-server')

        # correlate TF and TF model server versions
        for pkg in n_info.release_packages_all.keys():
            if n_info.release_packages_all[pkg]['component'] == 'tensorflow':
                package_ver=n_info.release_packages_all[pkg]['version']
                ver_digits = package_ver.rsplit('.')
                tf_small_ver=ver_digits[0]+'.'+ver_digits[1]
                for pkg2 in n_info.release_packages_all.keys():
                    if n_info.release_packages_all[pkg2]['component'] == 'tensorflow-model-server':
                        package_ver=n_info.release_packages_all[pkg2]['version']
                        ver_digits = package_ver.rsplit('.')
                        tf_model_server_small_ver=ver_digits[0]+'.'+ver_digits[1]
                        if tf_model_server_small_ver==tf_small_ver:
                            n_info.release_tf_package_to_model_server_package[pkg]=pkg2
                            break
        
        nr_setup.releases_info[neuron_release_ver]=n_info


    try:
        m_release=neuron_releases["neuron_versions"][neuron_version]["components"]
    except:
        print(__name__,": error: ","Version " + neuron_version + " is not a Neuron version or it is not supported")
        exit(-1)



    
    return (neuron_version,latest_neuron_version)





################
# Sanity Checks
################
def cli_validate(update,neuron_version,framework_version,is_latest_neuron,ami):
    # --update_cmd Sanity check
    # When choosing update, it always updating to latest , should not provide neuron_version
    if (update!=None) & (is_latest_neuron == False):
        print (__name__,": error: ","--update always update to latest Neuron versions, can't specify Neuron version")
        exit(-1)

    #if neuron_version != None:
    #    if ami == 'dlami':
    #        print (__name__,": error: ","--neuron_version should not be specified together with --ami=dlami")
    #        exit(-1)

    if (framework_version != None):
        if (framework_version not in  nr_setup.releases_info[neuron_version].release_frameworks_list):
            print (__name__,": error: "," " + framework_version + " is not a supported framework")
            exit(-1)

########################################
# version to tuple
########################################

def versiontuple(v):
   filled = []
   for point in v.split("."):
      filled.append(point.zfill(8))
   return tuple(filled)


########################################
# --list command
########################################
def cli_list_cmd(nr_setup, neuron_version, list):


    str =''

    if (list == 'neuron_versions'):
        str += '\nList of Neuron release versions supported by this helper:\n' + '\n'
        for ver in neuron_ver_list:
            str += 'neuron-'+ver + '\n'

    #TODO: add "[main]" label to main packages
    if (list == 'packages'):
        str += '\nList of Neuron packages included in Neuron release version ' + neuron_version + ':\n' + '\n'
        for package in nr_setup.releases_info[neuron_version].release_packages_all:
            if len( nr_setup.releases_info[neuron_version].release_packages_all[package]['package_type']):
                #FIXME Runtime library hardcode print
                if (nr_setup.releases_info[neuron_version].release_packages_all[package]["component"] == 'libnrt'):
                    str += nr_setup.releases_info[neuron_version].release_packages_all[package]["component"] +' : \t' +     \
                        "libnrt.so (version "+  \
                        nr_setup.releases_info[neuron_version].release_packages_all[package]["version"] +  ")"  + '\n'
                else:
                    str += nr_setup.releases_info[neuron_version].release_packages_all[package]["component"] +' : \t' + package + '\n'

    if (list == 'components'):
        str += '\nList of Neuron components included in Neuron release version ' + neuron_version + ':\n' + '\n'
        for comp in nr_setup.releases_info[neuron_version].release_components_list:
            str += comp + '\n'

    #TODO: add "[main]" label to main frameworks
    if (list == 'frameworks'):
        str += '\nList of frameworks included in Neuron release version ' + neuron_version + ':\n' + '\n'
        for fw in nr_setup.releases_info[neuron_version].release_frameworks_all:
            str += nr_setup.releases_info[neuron_version].release_frameworks_all[fw]["framework"] +' : \t' + fw + '\n'

    return str


########################################
# Print configuration
########################################

def hlpr_print_config(nr_setup, neuron_version):
    str = ''
    str += '\n'
    str += '###########################################################################' + '\n'
    str += '# ' + nr_setup.action + ' ' + nr_setup.framework + ' '
    if (nr_setup.framework_version != 'latest') & (nr_setup.framework_version != None):
        str += '(' + nr_setup.framework_version + ')' + ' '
    if nr_setup.action == 'Update':
        str += 'from latest Neuron version ' + neuron_version
    else: 
        str += 'from Neuron version ' + neuron_version
    
    str += '\n# '

    str += 'On '
    if (nr_setup.os == 'ubuntu'):
        str += 'Ubuntu '
    elif (nr_setup.os == 'amazonlinux'):
        str += 'Amazon Linux '

    if (nr_setup.ami == 'dlami'):
       str += 'DLAMI'
    else:
        str += 'AMI'

    str += ' for '
    if (nr_setup.mode == 'compile'):
       str += 'compilation on compute instance'
    elif (nr_setup.mode == 'develop'):
       str += 'development on inf1 instance'
    elif (nr_setup.mode == 'deploy'):
       str += 'deployment on inf1 instance'
    str += '\n'
    str += '###########################################################################' + '\n'
    str += '\n'

    return str

###################################
# Build Pip command
###################################
def hlpr_build_pip_command(nr_setup, neuron_version, component,include_compiler,optional):


    package_dict= nr_setup.releases_info[neuron_version].release_package_main

    if (nr_setup.framework_version==None):
        fw_package_dict= nr_setup.releases_info[neuron_version].release_frameworks_main
        fw_comp=component
    else:
        fw_package_dict= nr_setup.releases_info[neuron_version].release_frameworks_all
        fw_comp=nr_setup.framework_version
   
    pip_cmd_prefix=''
    pip_cmd =''


    if nr_setup.action=='Install':
        pip_cmd_prefix = 'pip install '
    else:
        pip_cmd_prefix = 'pip install --upgrade '

    cmd=pip_cmd_prefix

    if (component == 'mxnet') | (component == 'pytorch') | (component == 'tensorflow'):

        # Framework installation
        if (component == 'mxnet') | (component == 'pytorch'):
            pip_cmd += cmd + fw_package_dict[fw_comp]['package']
            if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                pip_cmd += '=='+fw_package_dict[fw_comp]['version']
            elif (nr_setup.is_latest_neuron==True)&(nr_setup.framework_version!=None):
                pip_cmd += '=='+fw_package_dict[fw_comp]['framework_version']+'.*'

        elif (component == 'tensorflow'):
            if ((parse(neuron_version)<parse('1.15.0')) | (parse(fw_package_dict[fw_comp]['framework_version'])<parse('2.0.0'))):
                pip_cmd += cmd + fw_package_dict[fw_comp]['package']
            else:
                pip_cmd = cmd + fw_package_dict[fw_comp]['package']
                if (include_compiler == True):
                    pip_cmd +=  '[cc]'
                    
            if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                pip_cmd += '=='+fw_package_dict[fw_comp]['version']
            elif (nr_setup.is_latest_neuron==True)&(nr_setup.framework_version!=None):
                pip_cmd += '=='+fw_package_dict[fw_comp]['framework_version']+'.*'

        # Compiler installation
        if (include_compiler == True):
            if (component == 'tensorflow'):
                if ((parse(neuron_version)<parse('1.15.0')) | (parse(fw_package_dict[fw_comp]['framework_version'])<parse('2.0.0'))):
                    pip_cmd += ' ' + package_dict['compiler']['package']
                    if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                        pip_cmd += '=='+package_dict['compiler']['version']
            if (component == 'mxnet'):
                pip_cmd += ' ' + package_dict['compiler']['package']
                if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                    pip_cmd += '=='+package_dict['compiler']['version']

            if (component == 'pytorch'):
                pip_cmd += ' ' + package_dict['compiler']['package']
                pip_cmd += '[tensorflow] "protobuf==3.20.1"'
                if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                    pip_cmd += '=='+package_dict['compiler']['version']

        # Additional packages installation
        if (component == 'pytorch'):
                pip_cmd += ' torchvision'

        if component == 'tensorflow':
            pip_cmd += ' "protobuf"'

    else:
        pip_cmd += '\n'
        if optional==False:
            pip_cmd += '# ' + nr_setup.action  + ' ' + package_formal_name[component] + '\n'
        else:
            pip_cmd += '# Optional: ' + nr_setup.action  + ' ' + package_formal_name[component] + '\n'
        pip_cmd += cmd + package_dict[component]['package']
        if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
            pip_cmd += '=='+package_dict[component]['version']




    pip_cmd += '\n'
    return pip_cmd




#################################################
##  pip_setup_repos
#################################################
def hlpr_pip_repos_setup():
    str = '\n'
    str += '# Set Pip repository  to point to the Neuron repository' + '\n'
    str += 'pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com'+ '\n'
    return str

#################################################
##  hlpr_pip_install_create_python_venv
#################################################

def hlpr_pip_install_create_python_venv(nr_setup, neuron_version):

    py_ver=nr_setup.releases_info[neuron_version].python_ver
    str = ''
    str += '\n'

    if nr_setup.os == 'ubuntu':        
        str += '######################################################' + '\n'
        str += '#   Only for Ubuntu 20 - Install Python' + py_ver + '\n'
        str += '#' + '\n'
        str += '# sudo add-apt-repository ppa:deadsnakes/ppa' + '\n'
        str += '# sudo apt-get install python' + py_ver + '\n'
        str += '#' + '\n'
        str += '######################################################' + '\n'

    str += '# Install Python venv and activate Python virtual environment to install    ' + '\n'
    str += '# Neuron pip packages.' + '\n'

    if nr_setup.os == 'ubuntu':        
        str += 'sudo apt-get install -y python'+ py_ver + '-venv g++' + '\n'        
    elif nr_setup.os == 'amazonlinux':
        str += 'sudo yum install -y python'+ py_ver + '-venv gcc-c++' + '\n'
    str += 'python'+ py_ver + ' -m venv ' + nr_setup.framework +'_venv' + '\n'
    str += 'source '+ nr_setup.framework  + '_venv/bin/activate' + '\n'
    str += 'pip install -U pip' + '\n'
    str += '\n'


    if (nr_setup.mode == 'develop') & (nr_setup.action =='Install'):
        if ((nr_setup.ami=='dlami') & (nr_setup.conda_env == 'None')) | \
            (nr_setup.ami !='dlami'):
        
            str += '\n'
            str += '# Instal Jupyter notebook kernel '+ '\n'
            str += 'pip install ipykernel ' + '\n'
            str += 'python'+ py_ver + ' -m ipykernel install --user --name '
            str += nr_setup.framework  + '_venv '
            str += '--display-name "Python (' + package_formal_name[nr_setup.framework] + ')"' + '\n' 
            str += 'pip install jupyter notebook' + '\n'
            str += 'pip install environment_kernels' + '\n'  
            str += '\n'

    return str

#################################################
##  hlpr_pip_activate_python_venv
#################################################

def hlpr_pip_activate_python_venv(nr_setup, neuron_version):

    py_ver=nr_setup.releases_info[neuron_version].python_ver

    str = ''
    str += '\n'
    str += '# Activate a Python ' + py_ver + ' virtual environment where Neuron pip packages were installed ' + '\n'
    str += 'source '+ nr_setup.framework  + '_venv/bin/activate' + '\n'
    str += '\n'
 
    return str

######################################################################
##  Framework/Compiler installation / Update  instructions (non-DLAMI)
#######################################################################

def hlpr_framework_compiler_setup(nr_setup, neuron_version, include_compiler):

    cmd_inst = ''
    cmd_inst += '\n'
    cmd_inst += '#' + nr_setup.action  + ' ' + package_formal_name[nr_setup.framework] + '\n'

    if (nr_setup.action=='Install'):
        if len(nr_setup.fw_package_dict[nr_setup.fw_comp]['pre_install_cmds']):
            for cmd_pre in nr_setup.releases_info[neuron_version].release_package_main[nr_setup.framework]['pre_install_cmds']:
                cmd_inst += cmd_pre  + '\n'

   
    cmd_inst += hlpr_build_pip_command(nr_setup=nr_setup,neuron_version=neuron_version, component=nr_setup.framework,include_compiler=include_compiler,optional=False) 

    return cmd_inst


######################################################################
##  hlpr_framework_dlami_activate
#######################################################################

def hlpr_framework_dlami_activate(nr_setup):

    str = ''

    str += '\n'
    if (nr_setup.framework == 'pytorch'):
            str += '# Activate PyTorch' + '\n'
    elif (nr_setup.framework == 'tensorflow'):
        str += '# Activate TensorFlow' + '\n'

    elif (nr_setup.framework == 'mxnet'): 
        str += '# Activate MXNet' + '\n'

    str += 'source activate '
    str +=  nr_setup.generic_conda_env + '\n'

    return str


#################################################
##  hlpr_os_packages_update
#################################################

def hlpr_os_packages_update(nr_setup):

    str = ''
    str += '\n'
    str += '# Update OS packages' + '\n'
    if nr_setup.os == 'ubuntu':
        str += 'sudo apt-get update -y' + '\n'
    elif nr_setup.os == 'amazonlinux':
        str += 'sudo yum update -y' + '\n'

    return str

#################################################
##  hlpr_os_headers_update
#################################################

def hlpr_os_headers_update(nr_setup):
    str = ''
    str = '\n'
    str += '# ' + nr_setup.action + ' OS headers'
    str += '\n'
    if nr_setup.os == 'ubuntu':
        str += 'sudo apt-get install linux-headers-$(uname -r) -y' + '\n' 
    elif nr_setup.os == 'amazonlinux':
        str += 'sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y' + '\n'
    return str 

#################################################
##  hlpr_os_export_path
#################################################

def hlpr_os_export_path(nr_setup):
    str = ''
    str += '\n'
    if nr_setup.os == 'ubuntu':
        str += 'export PATH=/opt/aws/neuron/bin:$PATH' + '\n'
    elif nr_setup.os == 'amazonlinux':
        str += 'export PATH=/opt/aws/neuron/bin:$PATH' + '\n'  
    return str


#################################################
##  hlpr_os_packages_first_setup
#################################################

def hlpr_os_packages_first_setup(nr_setup):

    str = ''
    str += '\n# Configure Linux for Neuron repository updates' + '\n'
    if nr_setup.os == 'ubuntu':
        str += '. /etc/os-release' + '\n'
        str += 'sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF' + '\n'
        str += 'deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main' + '\n'
        str += 'EOF' + '\n'
        str += 'wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -' + '\n'
    elif nr_setup.os == 'amazonlinux':
        str += 'sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF' + '\n'
        str += '[neuron]' + '\n'
        str += 'name=Neuron YUM Repository' + '\n'
        str += 'baseurl=https://yum.repos.neuron.amazonaws.com' + '\n'
        str += 'enabled=1' + '\n'
        str += 'metadata_expire=0' + '\n'
        str += 'EOF' + '\n'
        str += 'sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB' + '\n'

    return str


#################################################
##  os_comp_setup
#################################################

def hlpr_os_comp_setup_cmd(nr_setup, neuron_version, comp,optional,pkg):

    os_cmd = ''

    if pkg==None:
        key=comp
        pkg_dict= nr_setup.releases_info[neuron_version].release_package_main
    else:
        key=pkg
        pkg_dict= nr_setup.releases_info[neuron_version].release_packages_all




    if (comp=='driver'):    
        #os_cmd += '\n'
        #os_cmd += '###############################################################################################################\n'
        #os_cmd += '# Before installing or updating aws-neuron-dkms:'+ '\n'
        #os_cmd += '# - Stop any existing Neuron runtime 1.0 daemon (neuron-rtd) by calling: \'sudo systemctl stop neuron-rtd\'' + '\n'
        #os_cmd += '###############################################################################################################\n'
        # WARNING: Exception
        # Starting Neuron 1.16.0 , new kernel is needed to work with Runtime 2.x (library mode)


        if (parse(neuron_version)>=parse('2.99.99')):
            os_cmd += '\n'
            os_cmd += '################################################################################################################\n'
            os_cmd += '# To install or update to Neuron versions 2.99.99 and newer from previous releases:'+ '\n'
            if (nr_setup.os=='ubuntu'):
                os_cmd += '# - Unstall aws-neuron-dkms by calling \`sudo yum remove aws-neuron-dkms -y\`  -y'+ '\n'
            elif (nr_setup.os=='amazonlinux'):
                os_cmd += '# - Unstall aws-neuron-dkms by calling \`sudo apt-get remove aws-neuron-dkms\`  -y'+ '\n'
            os_cmd += '# - DO NOT skip \'aws-neuronx-dkms\' install or upgrade step, you MUST install or upgrade to latest Neuron driver'+ '\n'
            os_cmd += '################################################################################################################\n'
        elif (parse(neuron_version)>=parse('1.19.1')):
            os_cmd += '\n'
            os_cmd += '################################################################################################################\n'
            os_cmd += '# To install or update to Neuron versions 1.19.1 and newer from previous releases:'+ '\n'
            os_cmd += '# - DO NOT skip \'aws-neuron-dkms\' install or upgrade step, you MUST install or upgrade to latest Neuron driver'+ '\n'
            os_cmd += '################################################################################################################\n'


    # Update header files if driver should be installed or updated
    if (comp=='driver'):
        os_cmd += hlpr_os_headers_update(nr_setup)




    if nr_setup.os=='ubuntu':
        os_cmd_prefix = 'sudo apt-get install '
    elif (nr_setup.action=='Install')&(nr_setup.os=='amazonlinux'):
        os_cmd_prefix = 'sudo yum install '
    elif (nr_setup.action=='Update')&(nr_setup.os=='amazonlinux'):
        os_cmd_prefix = 'sudo yum update '

    if comp in nr_setup.releases_info[neuron_version].release_os_install_list:
        # install only if there is a package associated with the component
        if (len(pkg_dict[key]['package_type']) != 0):
            #os_cmd = build_os_command(cmd=os_cmd_prefix,component=comp,is_latest_release=is_latest_neuron)
            os_cmd += '\n' 
            if (optional==False):
                os_cmd += '# ' + nr_setup.action + ' ' + package_formal_name[comp]
            else:
                os_cmd += '# Optional: ' + nr_setup.action + ' ' + package_formal_name[comp]

            if (nr_setup.is_latest_neuron==False)&(nr_setup.os=='ubuntu'):
                os_cmd += '\n'
                os_cmd += '# If you are downgrading from newer version, please add \'--allow-downgrades\' option to \'sudo apt-get install\' '
            if (nr_setup.is_latest_neuron==False)&(nr_setup.os=='amazonlinux'):
                os_cmd += '\n'
                os_cmd += '# If you are downgrading from newer version , please remove existing package using \'sudo yum remove\' before installing the older package'
            os_cmd += '\n'
            # Amazon Linux DLAMI will not allow updating tensorflow-model-server and aws-neuron-dkms without adding sudo yum versionlock delete
            if ((comp=='tensorflow-model-server') | (comp=='driver'))  & (nr_setup.ami == 'dlami') & (nr_setup.os == 'amazonlinux'):
                os_cmd += 'sudo yum versionlock delete '
                os_cmd += pkg_dict[key]['package']
                os_cmd += '\n'

            os_cmd += os_cmd_prefix + pkg_dict[key]['package']

            # Amazon Linux yum installation packaging versioning is set via hyphen not equals
            version_key = "="
            if (nr_setup.os=='amazonlinux'):
                version_key = "-"

            if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions):
                os_cmd += version_key + pkg_dict[key]['version']
            elif (pkg!=None):
                if ( nr_setup.releases_info[neuron_version].release_package_main[comp]['version']!= nr_setup.releases_info[neuron_version].release_packages_all[pkg]['version']):
                    os_cmd += version_key + pkg_dict[key]['version']

            # Ubuntu DLAMI will not allow updating tensorflow-model-server and aws-neuron-dkms without adding --allow-change-held-packages
            if ((comp=='tensorflow-model-server') | (comp=='driver'))  & (nr_setup.ami == 'dlami') & (nr_setup.os == 'ubuntu'):
                os_cmd += ' --allow-change-held-packages'

            os_cmd += ' -y'
            os_cmd += '\n'

    # Update header files if driver should be installed or updated
    if (comp=='driver'):
        os_cmd += '\n'
        os_cmd += '####################################################################################\n'
        os_cmd += '# Warning: If Linux kernel is updated as a result of OS package update'+ '\n'
        if (parse(neuron_version)>=parse('2.99.99')):
            os_cmd += '#          Neuron driver (aws-neuronx-dkms) should be re-installed after reboot'+ '\n'
        else:
            os_cmd += '#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot'+ '\n'
        os_cmd += '####################################################################################\n'            

    if (comp=='tools'):
        if (parse(neuron_version)>=parse('2.99.99')):
            os_cmd += '\n'
            os_cmd += '################################################################################################################\n'
            os_cmd += '# To install or update to Neuron versions 2.99.99 and newer from previous releases:'+ '\n'
            if (nr_setup.os=='ubuntu'):
                os_cmd += '# - Unstall aws-neuron-tools by calling \`sudo yum remove aws-neuron-tools -y\`  -y'+ '\n'
            elif (nr_setup.os=='amazonlinux'):
                os_cmd += '# - Unstall aws-neuron-tools by calling \`sudo apt-get remove aws-neuron-tools\`  -y'+ '\n'
            os_cmd += '################################################################################################################\n'            

    return os_cmd


########################################
##  installation / Update  instructions
########################################
def hlpr_instructions(nr_setup, neuron_version):
    
    cmd_string = ''

    setup_mode=nr_setup.mode



    # look for conda environment for this framework version
    for fw_env in nr_setup.dlami_conda_env:
        if fw_env != nr_setup.framework:
            continue
        fw_ver_conda_env=nr_setup.dlami_conda_env[fw_env]
        for conda_env_fw_ver in fw_ver_conda_env:
            if (conda_env_fw_ver == nr_setup.fw_package_dict[nr_setup.fw_comp]['framework_version']):
                nr_setup.conda_env=nr_setup.dlami_conda_env[fw_env][conda_env_fw_ver][0]
                nr_setup.generic_conda_env=nr_setup.dlami_conda_env[fw_env][conda_env_fw_ver][1]
                break


    # look what runtime works with this framework version     
    fal_rtd=False
    fal_libnrt=False
    for fw in nr_setup.fal_supported_runtime:
        if fw != nr_setup.framework:
            continue
        if fw == nr_setup.framework:
            if (nr_setup.framework_version == None):
                fw_ver= nr_setup.releases_info[neuron_version].release_frameworks_main[nr_setup.framework]['framework_version']
                fal_version= nr_setup.releases_info[neuron_version].release_frameworks_main[nr_setup.framework]['version']
            else:
                fw_ver= nr_setup.releases_info[neuron_version].release_frameworks_all[nr_setup.framework_version]['framework_version']
                fal_version= nr_setup.releases_info[neuron_version].release_frameworks_all[nr_setup.framework_version]['version']
            fal_supported_rtd=nr_setup.fal_supported_runtime[fw][fw_ver]['neuron-rtd']
            fal_supported_libnrt=nr_setup.fal_supported_runtime[fw][fw_ver]['libnrt']            
            if (parse(fal_version) >= parse(fal_supported_rtd[0])) &  \
                (parse(fal_version) <= parse(fal_supported_rtd[1])):
                fal_rtd=True
            elif (parse(fal_version) >= parse(fal_supported_libnrt[0])) &  \
                (parse(fal_version) <= parse(fal_supported_libnrt[1])):
                fal_libnrt=True

    if nr_setup.conda_env == "None":
        dlami_ev_exists=False
    else:
        dlami_ev_exists=True

    #cmd_string += hlpr_print_config(nr_setup, neuron_version)

    if (nr_setup.framework_version==None):
        fw_package_dict= nr_setup.releases_info[neuron_version].release_frameworks_main
        fw_comp=nr_setup.framework
    else:
        fw_package_dict= nr_setup.releases_info[neuron_version].release_frameworks_all
        fw_comp=nr_setup.framework_version



    if (nr_setup.framework !=None): #if install or update
        # If we are not using DLAMI
        if (nr_setup.ami=='non-dlami') | \
            ((nr_setup.ami=='dlami') & \
                (
                (nr_setup.action == 'Update') | \
                (dlami_ev_exists==False) | \
                (nr_setup.is_latest_neuron==False)) \
                ):
         
           

            if (nr_setup.ami=='dlami') & (dlami_ev_exists==False):
                cmd_string += '\n'
                cmd_string += '# Note: There is no DLAMI Conda environment for this framework version'+ '\n'
                cmd_string += '#       Framework will be installed/updated inside a Python environment'+ '\n'


            if (setup_mode == 'develop') | (setup_mode == 'deploy'):
                if (nr_setup.action =='Install')&(nr_setup.ami!='dlami'):
                    # For First install, setup Neuron OS packagaes repo (yum or apt)
                    cmd_string += hlpr_os_packages_first_setup(nr_setup)

                # Always update to latest OS packages
                cmd_string += hlpr_os_packages_update(nr_setup)

                cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version, comp='driver',optional=False,pkg=None)


                #FIXME Temporary check for MXNET 1.5 in maintenance mode
                if (neuron_version == "1.16.0") & (nr_setup.framework=="mxnet")&    \
                    (fw_package_dict[fw_comp]['framework_version']=="1.5.1"):
                    cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version="1.15.2", comp='runtime-server',optional=False,pkg=None)
                elif (fal_rtd):
                    cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version, comp='runtime-server',optional=False,pkg=None)

                #if mode = develop, install tools
                if (setup_mode == 'develop'):
                    cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version, comp='tools',optional=False,pkg=None)
                    if (nr_setup.framework == 'tensorflow'):
                        cmd_string +=  hlpr_build_pip_command(nr_setup, neuron_version, component='tensorboard',include_compiler=False,optional=False)
            
                if (nr_setup.action =='Install'):
                    cmd_string += hlpr_os_export_path(nr_setup)

            if (nr_setup.ami=='non-dlami') | \
                ((nr_setup.ami=='dlami')&(nr_setup.generic_conda_env=="None")):

                if (nr_setup.action =='Install'):
                    # For first install , install python venv and activate a venv
                    cmd_string += hlpr_pip_install_create_python_venv(nr_setup, neuron_version)
                elif (nr_setup.action =='Update'):
                    # For nect times, activate the venv used for initial install
                    cmd_string += hlpr_pip_activate_python_venv(nr_setup, neuron_version)
            elif (nr_setup.ami=='dlami'):
                cmd_string += hlpr_framework_dlami_activate(nr_setup)
                
            # Setup Neuron pip packages
            cmd_string += hlpr_pip_repos_setup()
            
            # Now install framework
            if (setup_mode == 'deploy'):
                # do not install compiler when deploying
                cmd_string += hlpr_framework_compiler_setup(nr_setup, neuron_version,  include_compiler=False)
            else:
                # install compiler when mode = developer or mode = compile
                cmd_string += hlpr_framework_compiler_setup(nr_setup, neuron_version,  include_compiler=True)
        

            #if mode = deploy, install model server
            if (setup_mode != 'compile'):
                    if (nr_setup.framework == 'tensorflow'):
                        if (nr_setup.framework_version==None):
                            tf_package= nr_setup.releases_info[neuron_version].release_frameworks_main[nr_setup.framework]['package_name']
                        else:
                            tf_package= nr_setup.releases_info[neuron_version].release_frameworks_all[nr_setup.framework_version]['package_name']                         
                        cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version, comp='tensorflow-model-server',optional=True,pkg= nr_setup.releases_info[neuron_version].release_tf_package_to_model_server_package[tf_package])


        # if running DLAMI
        elif (nr_setup.ami=='dlami'):
            if (nr_setup.action =='Install'):

                cmd_string += '\n'
                cmd_string += '# Neuron is pre-installed on Deep Learning AMI (DLAMI), latest DLAMI version may not include latest Neuron versions '+ '\n'
                cmd_string += '# To update to latest Neuron version, follow "Update to latest release" instruction on Neuron documentation'+ '\n'

                # WARNING: Exception
                # Starting Neuron 1.16.0 , new kernel is needed to work with Runtime 2.x (library mode)
                if (parse(neuron_version)>=parse('1.16.0')):
                    if (setup_mode == 'develop') | (setup_mode == 'deploy'):
                        cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version, comp='driver',optional=False,pkg=None)

                #FIXME Temporary check for MXNET 1.5 in maintenance mode
                if (neuron_version == "1.16.0") & (nr_setup.framework=="mxnet")&    \
                    (fw_package_dict[fw_comp]['framework_version']=="1.5.1"):
                    cmd_string += hlpr_os_comp_setup_cmd(nr_setup, neuron_version="1.15.2", comp='runtime-server',optional=False,pkg=None)

                cmd_string += '\n'
                cmd_string += hlpr_framework_dlami_activate(nr_setup)


 
    return cmd_string





########################################
# neuron_setup_helper
########################################

class neuron_setup_helper:
    def __init__(self, manifest_file,neuron_version):

        # All Neuron releases
        self.releases_info = {}

        if (manifest_file== None) | (manifest_file== 'default')  :
            self.file = 'neuron-releases-manifest.json'
        else:
            self.file = manifest_file 

        ver_tuple = enumerate_release_manifest(nr_setup=self,in_neuron_version=neuron_version)
        self.neuron_version = ver_tuple[0]
        self.latest_neuron_version = ver_tuple[1]

        self.conda_env=""
        self.python_ver=""
        self.generic_conda_env=""
        
        if self.neuron_version == self.latest_neuron_version:
            self.is_latest_neuron=True
        else:
            self.is_latest_neuron=False

        if (self.is_latest_neuron) & (neuron_version !=None) & (neuron_version !='latest'):
            # User explicitly specified the version, although it is the latest version
            # in this case the instructions will include the exact versions of the packages
            self.force_versions=True
        else:
            self.force_versions=False


    def instructions(self,framework,action,framework_version,os,ami,mode):

        self.framework=framework
        self.action=action
        self.mode=mode
        self.os=os
        self.ami=ami
        if (framework_version=='latest'):
            self.framework_version=None
        else:
            self.framework_version=framework_version
        setup_cmd = ""

        if (self.framework_version==None):
            self.fw_package_dict= self.releases_info[self.neuron_version].release_frameworks_main
            self.fw_comp=self.framework
        else:
            self.fw_package_dict= self.releases_info[self.neuron_version].release_frameworks_all
            self.fw_comp=self.framework_version            

        setup_cmd=hlpr_instructions(self,self.neuron_version)
        
        return setup_cmd

if __name__ == '__main__':
    setup_cmd =''
    args = cli_parse_arguments()
    nr_setup=neuron_setup_helper(manifest_file=args.file,neuron_version=args.neuron_version)

    cli_validate(update=args.update,neuron_version=nr_setup.neuron_version,framework_version=args.framework_version,is_latest_neuron=nr_setup.is_latest_neuron,ami=args.ami)
    if (args.list):
        setup_cmd += cli_list_cmd(nr_setup=nr_setup,neuron_version=nr_setup.neuron_version, list=args.list)
    else:
        if (args.install != None)|(args.update !=None):    
            if args.install:
                framework=args.install
                action = 'Install'
            elif args.update:
                framework=args.update
                action = 'Update'
        else:
            action = None
            framework=None

        setup_cmd += nr_setup.instructions(framework=framework,action=action,framework_version=args.framework_version,os=args.os,ami=args.ami,mode=args.mode)
    print (setup_cmd)
    


    


