import json
import argparse
from packaging.version import Version, parse



# release_frameworks_all
# Desc: Dictionary - all framewors included in the release
#   example: 'pytorch-1.5.1': {'framework': 'pytorch', 'package': 'torch-neuron', 'version': '1.5.1.1.5.3.0', 'main': False, 'framework_version': '1.5.1', 'package_name': 'torch-neuron-1.5.1.1.5.3.0', 'pre_install_cmds': [], 'post_install_cmds': []}
release_frameworks_all = {}

# release_frameworks_main
# Desc: Dictionary - the main frameworks in each rlease (single  version of the same framework)
#   example: 'mxnet': {'framework': 'mxnet-1.8.0', 'package': 'mx_neuron', 'version': '1.8.0.1.3.0.0', 'framework_version': '1.5.1', 'full_package_name': 'mx_neuron-1.8.0.1.3.0.0', 'pre_install_cmds': ['wget https://aws-mx-pypi.s3-us-west-2.amazonaws.com/1.8.0/aws_mx_cu110-1.8.0-py2.py3-none-manylinux2014_x86_64.whl', 'pip install aws_mx_cu110-1.8.0-py2.py3-none-manylinux2014_x86_64.whl'], 'post_install_cmds': []}
release_frameworks_main = {}

# release_packages_all
# Desc: Dictionary -  all packages included in the release
#   example: 'aws-neuron-dkms-1.5.0.0': {'component': 'driver', 'package': 'aws-neuron-dkms', 'version': '1.5.0.0', 'main': True, 'pre_install_cmds': [], 'post_install_cmds': []}
release_packages_all ={}

# release_package_main
# Desc: Dictionary - only single package from each component
#   example: 'driver': {'package': 'aws-neuron-dkms', 'version': '1.5.0.0', 'full_package_name': 'aws-neuron-dkms-1.5.0.0', 'pre_install_cmds': [], 'post_install_cmds': []}
release_package_main={}

# list of all Neuron versions included in the manifest
neuron_ver_list = []      

# list of all framewoks included in the specific neuron release
release_frameworks_list=[]

# list of all neuron components included in the specific neuron release
release_components_list = []

# dictionary to correlate tf version with model server version
tf_package_to_model_server_package = {}




os_install_list =[]

package_formal_name= {
    "compiler":"Neuron Compiler",
    "tensorflow":"Neuron TensorFlow",
    "pytorch":"Neuron PyTorch",
    "mxnet":"Neuron MXNet",
    "runtime":"Neuron Runtime",
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




def enumerate_release_manifest(file,in_neuron_version):

    ########################################
    # Enumerate the Json file
    ########################################

    if file==None:
        file='neuron-releases-manifest.json'

    try:
        read_file = open(file, "r") 
    except:
        print(__name__,": error:","Can't open " + file + " ")
        exit(-1)

    neuron_releases = json.load (read_file)

    latest_neuron_version = neuron_releases["latest_release"]["inf1"]["version"]

    if (in_neuron_version == None) | (in_neuron_version == 'latest'):
        neuron_version=latest_neuron_version
    else:
        neuron_version = in_neuron_version


    for n_ver in neuron_releases["neuron_versions"]:
        neuron_ver_list.append(n_ver)

    try:
        m_release=neuron_releases["neuron_versions"][neuron_version]["components"]
    except:
        print(__name__,": error: ","Version " + neuron_version + " is not a Neuron version or it is not supported")
        exit(-1)

    for component_name in m_release:
        if m_release[component_name]["framework"]==False:
            release_components_list.append(component_name)    
        m_packages=m_release[component_name]["packages"]
        for package_name in m_packages:
            for package_ver in m_packages[package_name]["versions"]:
                m_package_ver=m_packages[package_name]["versions"][package_ver]

                full_package_name=package_name+'-'+package_ver

                release_packages_all[full_package_name]= {"component":component_name,"package":package_name,"version":package_ver,"main":m_package_ver["main_version"],"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}

                if m_package_ver["main_version"]:
                    release_package_main[component_name]={"package":package_name,"version":package_ver,"full_package_name":full_package_name,"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}

                if m_release[component_name]["framework"]:
                    ver_digits = package_ver.rsplit('.')
                    fw_ver=ver_digits[0]+'.'+ver_digits[1]+'.'+ver_digits[2]
                    fw_name_ver=component_name+'-'+fw_ver

                    if m_release[component_name]["framework"]:
                        release_components_list.append(fw_name_ver)
                        release_frameworks_list.append(fw_name_ver)

                    if m_package_ver["main_version"]:
                        release_frameworks_main[component_name]={"framework":fw_name_ver,"package":package_name,"version":package_ver,"framework_version":fw_ver,"package_name":full_package_name,"full_package_name":full_package_name,"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}


                    release_frameworks_all[fw_name_ver]={"framework":component_name,"package":package_name,"version":package_ver,"main":m_package_ver["main_version"],"framework_version":fw_ver,"package_name":full_package_name,"pre_install_cmds":m_package_ver["pre_install_cmds"],"post_install_cmds":m_package_ver["post_install_cmds"],"package_type":m_package_ver["package_type"]}

    if 'driver' in release_components_list:
        os_install_list.append('driver')
    if 'runtime' in release_components_list:
        os_install_list.append('runtime')
    if 'tools' in release_components_list:
        os_install_list.append('tools')
    if 'tensorflow-model-server' in release_components_list:
        os_install_list.append('tensorflow-model-server')


    # correlate TF and TF model server versions
    for pkg in release_packages_all.keys():
        if release_packages_all[pkg]['component'] == 'tensorflow':
            package_ver=release_packages_all[pkg]['version']
            ver_digits = package_ver.rsplit('.')
            tf_small_ver=ver_digits[0]+'.'+ver_digits[1]
            for pkg2 in release_packages_all.keys():
                if release_packages_all[pkg2]['component'] == 'tensorflow-model-server':
                    package_ver=release_packages_all[pkg2]['version']
                    ver_digits = package_ver.rsplit('.')
                    tf_model_server_small_ver=ver_digits[0]+'.'+ver_digits[1]
                    if tf_model_server_small_ver==tf_small_ver:
                        tf_package_to_model_server_package[pkg]=pkg2
                        break
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
        if (framework_version not in release_frameworks_list):
            print (__name__,": error: "," " + framework_version + " is not a supported framework")
            exit(-1)






########################################
# --list command
########################################
def cli_list_cmd(nr_setup,list):


    str =''

    if (list == 'neuron_versions'):
        str += '\nList of Neuron release versions supported by this helper:\n' + '\n'
        for ver in neuron_ver_list:
            str += 'neuron-'+ver + '\n'

    #TODO: add "[main]" label to main packages
    if (list == 'packages'):
        str += '\nList of Neuron packages included in Neuron release version ' + nr_setup.neuron_version + ':\n' + '\n'
        for package in release_packages_all:
            str += release_packages_all[package]["component"] +' : \t' + package + '\n'

    if (list == 'components'):
        str += '\nList of Neuron components included in Neuron release version ' + nr_setup.neuron_version + ':\n' + '\n'
        for comp in release_components_list:
            str += comp + '\n'

    #TODO: add "[main]" label to main frameworks
    if (list == 'frameworks'):
        str += '\nList of frameworks included in Neuron release version ' + nr_setup.neuron_version + ':\n' + '\n'
        for fw in release_frameworks_all:
            str += release_frameworks_all[fw]["framework"] +' : \t' + fw + '\n'

    return str


########################################
# Print configuration
########################################

def hlpr_print_config(nr_setup):
    str = ''
    str += '\n'
    str += '###########################################################################' + '\n'
    str += '# ' + nr_setup.action + ' ' + nr_setup.framework + ' '
    if (nr_setup.framework_version != 'latest') & (nr_setup.framework_version != None):
        str += '(' + nr_setup.framework_version + ')' + ' '
    if nr_setup.action == 'Update':
        str += 'from latest Neuron version ' + nr_setup.neuron_version
    else: 
        str += 'from Neuron version ' + nr_setup.neuron_version
    
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
def hlpr_build_pip_command(nr_setup,component,include_compiler,optional):


    package_dict=release_package_main

    if (nr_setup.framework_version==None):
        fw_package_dict=release_frameworks_main
        fw_comp=component
    else:
        fw_package_dict=release_frameworks_all
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
            if (parse(nr_setup.neuron_version)<parse('1.15.0')):
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
                if (Version(nr_setup.neuron_version)<Version('1.15.0')):
                    pip_cmd += ' ' + package_dict['compiler']['package']
                    if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                        pip_cmd += '=='+package_dict['compiler']['version']
            if (component == 'mxnet'):
                pip_cmd += ' ' + package_dict['compiler']['package']
                if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                    pip_cmd += '=='+package_dict['compiler']['version']

            if (component == 'pytorch'):
                pip_cmd += ' ' + package_dict['compiler']['package']
                pip_cmd += '[tensorflow]'
                if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions == True):
                    pip_cmd += '=='+package_dict['compiler']['version']

        # Additional packages installation
        if (component == 'pytorch'):
                pip_cmd += ' torchvision'


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

def hlpr_pip_install_create_python_venv(nr_setup):
    str = ''
    str += '\n'
    str += '# Install Python venv and activate Python virtual environment to install    ' + '\n'
    str += '# Neuron pip packages.' + '\n'

    if nr_setup.os == 'ubuntu':
        str += 'sudo apt-get install -y python3-venv g++' + '\n'
    elif nr_setup.os == 'amazonlinux':
        str += 'sudo yum install -y python3 gcc-c++' + '\n'
    str += 'python3 -m venv ' + nr_setup.framework +'_venv' + '\n'
    str += 'source '+ nr_setup.framework  + '_venv/bin/activate' + '\n'
    str += 'pip install -U pip' + '\n'
    str += '\n'
    return str

#################################################
##  hlpr_pip_activate_python_venv
#################################################

def hlpr_pip_activate_python_venv(nr_setup):
    str = ''
    str += '\n'
    str += '# Activate Python virtual environment where Neuron pip packages were installed ' + '\n'
    str += 'source '+ nr_setup.framework  + '_venv/bin/activate' + '\n'
    str += '\n'
    return str

######################################################################
##  Framework/Compiler installation / Update  instructions (non-DLAMI)
#######################################################################

def hlpr_framework_compiler_setup(nr_setup,include_compiler):

    cmd_inst = ''
    cmd_inst += '\n'
    cmd_inst += '#' + nr_setup.action  + ' ' + package_formal_name[nr_setup.framework] + '\n'

    if (nr_setup.action=='Install'):
        if len(nr_setup.fw_package_dict[nr_setup.fw_comp]['pre_install_cmds']):
            for cmd_pre in release_package_main[nr_setup.framework]['pre_install_cmds']:
                cmd_inst += cmd_pre  + '\n'

   
    cmd_inst += hlpr_build_pip_command(nr_setup=nr_setup,component=nr_setup.framework,include_compiler=include_compiler,optional=False) 
            
    return cmd_inst


######################################################################
##  hlpr_framework_dlami_activate
#######################################################################

def hlpr_framework_dlami_activate(nr_setup):

    str = ''

    str += '\n'
    if (nr_setup.framework == 'pytorch'):
            str += '# Activate PyTorch' + '\n'
            str += 'source activate aws_neuron_pytorch_p36' + '\n'
    elif (nr_setup.framework == 'tensorflow'):
        str += '# Activate TensorFlow' + '\n'
        str += 'source activate aws_neuron_tensorflow_p36' + '\n'

    elif (nr_setup.framework == 'mxnet'): 
        str += '# Activate MXNet' + '\n'
        str += 'source activate aws_neuron_mxnet_p36' + '\n'

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

def hlpr_os_comp_setup_cmd(nr_setup,comp,optional,pkg):

    os_cmd = ''

    if pkg==None:
        key=comp
        pkg_dict=release_package_main
    else:
        key=pkg
        pkg_dict=release_packages_all
        
    # WARNING: Exception
    # Starting Neuron 1.16.0 , new kernel is needed to work with Runtime 2.x (library mode)
    if (comp=='driver') & (parse(nr_setup.neuron_version)>=parse('1.16.0')):
        os_cmd += '\n'
        os_cmd += '#################################################################################\n'
        os_cmd += '# Starting Neuron 1.16.0 - please make sure to update to latest Neuron driver'+ '\n'
        os_cmd += '#################################################################################\n'

    if nr_setup.os=='ubuntu':
        os_cmd_prefix = 'sudo apt-get install '
    elif (nr_setup.action=='Install')&(nr_setup.os=='amazonlinux'):
        os_cmd_prefix = 'sudo yum install '
    elif (nr_setup.action=='Update')&(nr_setup.os=='amazonlinux'):
        os_cmd_prefix = 'sudo yum update '

    if comp in os_install_list:
        # install only if there is a package associated with the component
        if (len(pkg_dict[key]['package_type']) != 0):
            #os_cmd = build_os_command(cmd=os_cmd_prefix,component=comp,is_latest_release=is_latest_neuron)
            os_cmd += '\n' 
            if (optional==False):
                os_cmd += '# ' + nr_setup.action + ' ' + package_formal_name[comp]
            else:
                os_cmd += '# Optional: ' + nr_setup.action + ' ' + package_formal_name[comp]
            os_cmd += '\n' 
            os_cmd += os_cmd_prefix + pkg_dict[key]['package']
            if (nr_setup.is_latest_neuron==False) | (nr_setup.force_versions):
                os_cmd += '=' + pkg_dict[key]['version']
            elif (pkg!=None):
                if (release_package_main[comp]['version']!=release_packages_all[pkg]['version']):
                    os_cmd += '=' + pkg_dict[key]['version']
            os_cmd += ' -y'
            os_cmd += '\n'
    
    return os_cmd


########################################
##  installation / Update  instructions
########################################
def hlpr_instructions(nr_setup):
    
    cmd_string = ''

    setup_mode=nr_setup.mode

    # WARNING: Exception
    # TensorFlow 2 still have no DLAMI conda env , so create or activate python venv
    # the below condition should be removed when DLAMI will add new DLAMI conda for TensorFlow 2    
    if ((nr_setup.ami=='dlami')&(parse(nr_setup.fw_package_dict[nr_setup.fw_comp]['framework_version']) >= parse('2.0.0'))):
        dlami_ev_exists=False
    else:
        dlami_ev_exists=True

    #cmd_string += hlpr_print_config(nr_setup)

    if (nr_setup.framework !=None): #if install or update
        # If we are not using DLAMI
        if (nr_setup.ami=='non-dlami') | \
            ((nr_setup.ami=='dlami') & \
                (
                (nr_setup.action == 'Update') | \
                (dlami_ev_exists==False) | \
                (nr_setup.framework_version!=None) | \
                (nr_setup.is_latest_neuron==False)) \
                ):
         
           
            if (setup_mode == 'develop') | (setup_mode == 'deploy'):
                if (nr_setup.action =='Install'):
                    # For First install, setup Neuron OS packagaes repo (yum or apt)
                    cmd_string += hlpr_os_packages_first_setup(nr_setup)

                # Always update to latest OS packages
                cmd_string += hlpr_os_packages_update(nr_setup)

                # Update header file
                if (nr_setup.action =='Install'):
                    cmd_string += hlpr_os_headers_update(nr_setup)

                cmd_string += hlpr_os_comp_setup_cmd(nr_setup,comp='driver',optional=False,pkg=None)
                cmd_string += hlpr_os_comp_setup_cmd(nr_setup,comp='runtime',optional=False,pkg=None)
            
                if (nr_setup.action =='Install'):
                    cmd_string += hlpr_os_export_path(nr_setup)

            if (nr_setup.ami=='non-dlami') | \
                ((nr_setup.ami=='dlami')&(dlami_ev_exists==False)):

                if (nr_setup.action =='Install'):
                    # For first install , install python venv and activate a venv
                    cmd_string += hlpr_pip_install_create_python_venv(nr_setup)
                elif (nr_setup.action =='Update'):
                    # For nect times, activate the venv used for initial install
                    cmd_string += hlpr_pip_activate_python_venv(nr_setup)
            elif (nr_setup.ami=='dlami'):
                cmd_string += hlpr_framework_dlami_activate(nr_setup)
                
            # Setup Neuron pip packages
            cmd_string += hlpr_pip_repos_setup()
            
            # Now install framework
            if (setup_mode == 'deploy'):
                # do not install compiler when deploying
                cmd_string += hlpr_framework_compiler_setup(nr_setup, include_compiler=False)
            else:
                # install compiler when mode = developer or mode = compile
                cmd_string += hlpr_framework_compiler_setup(nr_setup, include_compiler=True)
            
            #if mode = develop, install tools
            if (setup_mode == 'develop'):
                cmd_string += hlpr_os_comp_setup_cmd(nr_setup,comp='tools',optional=False,pkg=None)
                if (nr_setup.framework == 'tensorflow'):                    
                    cmd_string +=  hlpr_build_pip_command(nr_setup,component='tensorboard',include_compiler=False,optional=False)

            #if mode = deploy, install model server
            if (setup_mode != 'compile'):
                    if (nr_setup.framework == 'tensorflow'):
                        if (nr_setup.framework_version==None):
                            tf_package=release_frameworks_main[nr_setup.framework]['package_name']
                        else:
                            tf_package=release_frameworks_all[nr_setup.framework_version]['package_name']                         
                        cmd_string += hlpr_os_comp_setup_cmd(nr_setup,comp='tensorflow-model-server',optional=True,pkg=tf_package_to_model_server_package[tf_package])

            
        # if running DLAMI
        elif (nr_setup.ami=='dlami'):
            if (nr_setup.action =='Install'):
                cmd_string += '\n'
                cmd_string += '# Please note that latest DLAMI version may not include latest Neuron versions'+ '\n'
                cmd_string += '# Use update instructions to update to latest Neuron versions and install all components'+ '\n'
                cmd_string += '\n'

                cmd_string += hlpr_framework_dlami_activate(nr_setup)

                # WARNING: Exception
                # Starting Neuron 1.16.0 , new kernel is needed to work with Runtime 2.x (library mode)
                if (parse(nr_setup.neuron_version)>=parse('1.16.0')):
                    cmd_string += hlpr_os_comp_setup_cmd(nr_setup,comp='driver',optional=False,pkg=None)

 
    return cmd_string




########################################
# neuron_setup_helper
########################################

class neuron_setup_helper:
    def __init__(self, manifest_file,neuron_version):

        if (manifest_file== None) | (manifest_file== 'default')  :
            self.file = 'neuron-releases-manifest.json'
        else:
            self.file = manifest_file 

        ver_tuple = enumerate_release_manifest(file=self.file,in_neuron_version=neuron_version)
        self.neuron_version = ver_tuple[0]
        self.latest_neuron_version = ver_tuple[1]

        
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
            self.fw_package_dict=release_frameworks_main
            self.fw_comp=self.framework
        else:
            self.fw_package_dict=release_frameworks_all
            self.fw_comp=self.framework_version            

        setup_cmd=hlpr_instructions(self)
        
        return setup_cmd

if __name__ == '__main__':
    setup_cmd =''
    args = cli_parse_arguments()
    nr_setup=neuron_setup_helper(manifest_file=args.file,neuron_version=args.neuron_version)

    cli_validate(update=args.update,neuron_version=args.neuron_version,framework_version=args.framework_version,is_latest_neuron=nr_setup.is_latest_neuron,ami=args.ami)
    if (args.list):
        setup_cmd += cli_list_cmd(nr_setup=nr_setup,list=args.list)
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
    


    


