import json
import argparse
from packaging.version import Version, parse
import pandas as pd
from pandas import json_normalize


class manifest:
    def __init__(self, manifest_file):

        self.manifest_file = manifest_file
        self.df_packages = pd.DataFrame()

    def parse_manifest(self):

        with open(self.manifest_file, 'r') as f:
            manifest = json.load(f)

        # repos
        self.df_repos = json_normalize(manifest['repos_n2'])

        # latest release
        self.df_latest_release = json_normalize(manifest['latest_release'])

        # os properties
        self.df_os_properties = json_normalize(manifest['os_properties'])

        # ami properties
        self.df_ami_properties = json_normalize(manifest['ami_properties'])

        # dlami properties
        self.df_dlami_properties = json_normalize(manifest['dlami_properties'])

        # major version properties
        self.df_major_version_properties = json_normalize(manifest['major_version_properties'])

        # package properties
        self.df_package_properties = json_normalize(manifest['package_properties'])

        # neuron releases
        for release in manifest['neuron_releases']:
            df_release = json_normalize(release['packages'])
            df_release['neuron_version'] = release['neuron_version']
            self.df_packages = pd.concat([self.df_packages, df_release])

        # merge release packages
        self.df_release_packages = self.df_packages.merge(self.df_package_properties, how='left', on='name')
        self.df_release_packages['supported_instances'] = self.df_release_packages['supported_instances'].tolist()

    def merge_release_packages(self):

        self.df_release_packages = self.df_packages.merge(self.df_package_properties, how='left', on='name')

    def extract_major_minor_version(self, version):

        return str(version.major) + '.' + str(version.minor)

    def get_pip_packages_supporting_python_versions(self, args):
        '''
        Get supported python version by packages (compiler and framework)
        e.g., {"3.6","3.7","3.8"}
        '''

        if args.neuron_version == None:
            neuron_version = self.get_latest_neuron_version_per_instance(args.instance)
        else:
            neuron_version = args.neuron_version

        df_instance = self.df_release_packages[
            (self.df_release_packages['supported_instances'].map(lambda x: args.instance in x)) & (
                    self.df_release_packages['neuron_version'] == neuron_version)]

        # Compiler supporting Python versions
        compiler_python_versions = \
            df_instance.loc[df_instance['component'] == 'Compiler']['supported_python_versions'].values[0]

        # Specific framework version supporting Python versions
        df_framework = df_instance.loc[df_instance['category'] == args.framework].copy()
        df_framework['version'] = df_framework['version'].map(lambda x: Version(x))
        df_framework['major_minor_version'] = df_framework['version'].map(lambda x: str(x.major) + '.' + str(x.minor))

        framework_python_versions = df_framework.loc[
            df_framework['major_minor_version'] == self.extract_major_minor_version(Version(args.framework_version))][
            'supported_python_versions'].values[0]
        return list(set(compiler_python_versions) & set(framework_python_versions))

    def get_major_version(self, package_name, instance):
        return self.df_major_version_properties.loc[(self.df_major_version_properties['name'] == package_name)][
            args.instance].values[0]

    def generate_script(self, args):
        '''
        It generates:
        (1) str_preamble
        (2) str_driver
        (3) str_runtime
        (4) str_tools
        (5) str_python
        (6) str_compiler
        (7) str_framework
        '''

        str_preamble = ''

        # Install and enable EPEL (required only for rocky linux 9 currently)
        str_preamble += self.install_and_enable_epel(args)

        # Configure Neuron repository
        str_preamble += self.config_neuron_repository(args)

        # Update OS packages
        str_preamble += self.update_os_packages(args)

        # Install OS headers
        str_preamble += self.install_os_headers(args)

        # Install git
        str_preamble += self.install_git(args)

        # Install Neuron driver
        str_driver = self.install_neuron_driver(args)

        # Install Neuron runtime
        str_runtime = self.install_neuron_runtime(args)

        # Install EFA driver
        str_efa = self.install_efa_driver(args)

        # Install Neuron Tools
        str_tools = self.install_neuron_system_tools(args)

        # Add PATH
        if args.mode != 'compile' or args.ami != 'dlami-framework':
            str_tools += '\n# Add PATH\n'
            str_tools += 'export PATH=/opt/aws/neuron/bin:$PATH\n'

        # Install Python virtual environment
        str_python = self.set_python_venv(args)

        # Activate Pythohn venv
        str_python += self.activate_python_venv(args)

        # install jupyter notebook
        str_python += self.jupyter_notebook(args)

        # Set pip repository
        str_python += self.set_pip_repository()

        # Install wget, awscli
        str_python += self.install_aux(args)

        # install extra dependencies
        str_deps = self.install_extra_dependencies(args)

        # Install Neuron compiler
        str_compiler = self.install_neuron_compiler(args)

        # Install Neuron framework
        str_framework = self.install_neuron_framework(args)

        # install neuron compiler and framework
        str_compiler_framework = self.install_neuron_compiler_and_framework(args)
        if args.ami == 'dlami-framework':
            # dlami instructions
            str_dlami = self.install_dlami(args)
            return str_dlami
        elif args.ami == 'dlami-neuron':
            str_dlami = self.install_neuron_dlami(args)
            return str_dlami
        elif args.category == 'all':
            if args.instance == 'trn1':
                str_runtime += str_efa
            return str_preamble + str_driver + str_runtime + str_tools + str_deps + str_python + str_compiler_framework
        elif args.category == 'driver_runtime_tools':
            return str_preamble + str_driver + str_runtime + str_tools
        elif args.category == 'compiler_framework':
            return str_deps + str_python + str_compiler_framework
        elif args.category == 'driver':
            return str_preamble + str_driver
        elif args.category == 'runtime':
            return str_runtime
        elif args.category == 'tools':
            return str_tools
        elif args.category == 'compiler':
            if args.instance != 'inf1':
                return str_python + str_compiler
            else:
                return str_python
        elif args.category == 'framework':
            return str_framework
        elif args.category == 'efa':
            return str_efa

    def install_dlami(self, args):
        latest_release_for_instance = \
            self.df_latest_release.loc[self.df_latest_release['instance'] == args.instance]['version'].values[0]
        latest_release_for_dlami = self.df_dlami_properties[
            (self.df_dlami_properties['framework'] == args.framework) & (
                self.df_dlami_properties['supported_instances'].map(lambda x: args.instance in x))][
            'neuron_released_version'].values[0]

        if (latest_release_for_instance == latest_release_for_dlami):
            return self.activate_python_venv(args)
        else:
            args.install_type = 'update'
            str_dlami = self.activate_python_venv(args)
            str_dlami += self.jupyter_notebook(args)
            str_dlami += self.set_pip_repository()
            str_dlami += self.install_neuron_compiler_and_framework(args)
        return str_dlami


    def install_neuron_dlami(self, args):
        str_dlami = ""
        if ((args.instance == 'trn1' or args.instance == 'inf2') and args.category == "transformers-neuronx"):
            str_dlami = '\n# Activate Python venv for Transformers-NeuronX \n'
            str_dlami += "source /opt/aws_neuronx_venv_transformers_neuronx/bin/activate"
        elif ((args.instance == 'trn1' or args.instance == 'inf2') and args.framework == "pytorch" and args.framework_version == "1.13.1"):
            str_dlami = '\n# Activate Python venv for Pytorch 1.13 \n'
            str_dlami += "source /opt/aws_neuronx_venv_pytorch_1_13/bin/activate"
        elif ((args.instance == 'trn1' or args.instance == 'inf2') and args.framework == "pytorch" and args.framework_version == "2.1"):
            str_dlami = '\n# Activate Python venv for Pytorch 2.1 \n'
            str_dlami += "source /opt/aws_neuronx_venv_pytorch_2_1/bin/activate"
        elif ((args.instance == 'trn1' or args.instance == 'inf2') and args.framework == "tensorflow" and args.framework_version == "2.10.1"):
            str_dlami = '\n# Activate Python venv for Tensorflow 2.10 \n'
            str_dlami += "source /opt/aws_neuronx_venv_tensorflow_2_10/bin/activate"
        elif (args.instance == 'inf1' and args.framework == "tensorflow" and args.framework_version == "2.10.1"):
            str_dlami = '\n# Activate Python venv for Tensorflow 2.10 \n'
            str_dlami += "source /opt/aws_neuron_venv_tensorflow_2_10_inf1/bin/activate"
        elif (args.instance == 'inf1' and args.framework == "pytorch" and args.framework_version == "1.13.1"):
            str_dlami = '\n# Activate Python venv for Pytorch 1.13 \n'
            str_dlami += "source /opt/aws_neuron_venv_pytorch_1_13_inf1/bin/activate"
        return str_dlami

    def jupyter_notebook(self, args):
        os_default_python_version = \
            self.df_os_properties.loc[self.df_os_properties['os'] == args.os]['default_python_version'].values[0]
        packages_supporting_python_versions = self.get_pip_packages_supporting_python_versions(args)

        if os_default_python_version in packages_supporting_python_versions:
            target_python_version = os_default_python_version
        else:
            target_python_version = max(packages_supporting_python_versions)

        framework_name = self.get_package_names(category=args.framework, instance=args.instance)[0]

        str_jupiter = '\n# Install Jupyter notebook kernel\n'
        str_jupiter += 'pip install ipykernel ' + '\n'
        str_jupiter += 'python' + target_python_version + ' -m ipykernel install --user --name '
        str_jupiter += 'aws_neuron_venv_' + args.framework
        if args.instance == 'inf1':
            str_jupiter += '_inf1'
        str_jupiter += ' --display-name "Python (' + framework_name + ')"' + '\n'
        str_jupiter += 'pip install jupyter notebook' + '\n'
        str_jupiter += 'pip install environment_kernels' + '\n'
        return str_jupiter

    def install_and_enable_epel(self, args):
        str = ''
        if args.mode != 'compile':
            if args.install_type == 'install':
                if args.os == 'rockylinux9':
                    str += '\n# Install and enable EPEL\n'
                    str += 'sudo dnf config-manager --set-enabled crb\n'
                    str += 'sudo dnf install epel-release -y\n'
        return str

    def config_neuron_repository(self, args):
        """
        Reads OS type from the arguments and generates scripts for configuration of Neuron repository
        """
        str = ''
        if args.mode != 'compile':
            # Neuron repository needs when mode is 'develop' or 'deploy'
            if args.install_type == 'install':
                str += '\n# Configure Linux for Neuron repository updates' + '\n'
                if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                    str += '. /etc/os-release' + '\n'
                    str += 'sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF' + '\n'
                    str += 'deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main' + '\n'
                    str += 'EOF' + '\n'
                    str += 'wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -' + '\n'
                elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
                    str += 'sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF' + '\n'
                    str += '[neuron]' + '\n'
                    str += 'name=Neuron YUM Repository' + '\n'
                    str += 'baseurl=https://yum.repos.neuron.amazonaws.com' + '\n'
                    str += 'enabled=1' + '\n'
                    str += 'metadata_expire=0' + '\n'
                    str += 'EOF' + '\n'
                    str += 'sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB' + '\n'
        return str


    def get_repo(self):
        str = '\n# Configure Linux for Neuron repository updates' + '\n'
        if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
            str += '. /etc/os-release' + '\n'
            str += 'sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF' + '\n'
            str += 'deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main' + '\n'
            str += 'EOF' + '\n'
            str += 'wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -' + '\n'
        elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
            str += 'sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF' + '\n'
            str += '[neuron]' + '\n'
            str += 'name=Neuron YUM Repository' + '\n'
            str += 'baseurl=https://yum.repos.neuron.amazonaws.com' + '\n'
            str += 'enabled=1' + '\n'
            str += 'metadata_expire=0' + '\n'
            str += 'EOF' + '\n'
            str += 'sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB' + '\n'
        return str

    def update_os_packages(self, args):
        """
        Reads mode and OS type and updates OS packages accordingly.
        """
        str = ''
        if args.mode != 'compile':
            # OS packages need to be updated in "develop" or "deploy" mode
            str += '\n# Update OS packages \n'
            if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                str += 'sudo apt-get update -y' + '\n'
            elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
                str += 'sudo dnf update -y' + '\n'
            if args.os == 'rockylinux9':
                str += '# Reboot instance to ensure kernel is updated\n'
                str += 'sudo reboot\n'
        return str

    def install_os_headers(self, args):
        """
        Reads mode and OS type and install OS headers accordingly.
        """
        str = ''
        if args.mode != 'compile':
            # OS headers need to be installed in "develop" or "deploy" mode
            if args.install_type == 'install':
                str += '\n# Install OS headers \n'
            elif args.install_type == 'update':
                str += '\n# Update OS headers \n'
            if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                str += 'sudo apt-get install linux-headers-$(uname -r) -y' + '\n'
            elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
                str += 'sudo dnf install -y "kernel-devel-uname-r = $(uname -r)"' + '\n'

        return str

    def install_git(self, args):

        str = '\n# Install git \n'
        if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
            str += 'sudo apt-get install git -y\n'
        elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
            str += 'sudo dnf install git -y\n'

        return str

    def install_neuron_driver(self, args):
        """
        Neuron driver install script will be generated based on mode, AMI, and OS.
        mode: when develop or deploy
        AMI: when not dlami-base
        OS: for different command
        """
        str = ''

        if args.ami == 'dlami-base':
            return str

        # get driver package names for release version, instance
        # we take only the first element in the list since there should be ond driver package.
        driver_package = self.get_package_names(category='driver', instance=args.instance)[0]

        if args.mode != 'compile':
            # if args.ami != 'dlami-base':
            install = 'install' if args.install_type == 'install' else 'upgrade'
            str += f'\n# {install} Neuron Driver\n'

            if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                if args.neuron_version == None:
                    if self.df_package_properties.loc[self.df_package_properties['name'] == driver_package][
                        'pin_major'].values[0] == 'true':
                        version = '=' + self.get_major_version(driver_package, args.instance) + '.'
                elif (args.neuron_version != None) & (args.install_type == 'install'):
                    version = '=' + self.get_package_version(category='driver', name=driver_package,
                                                             neuron_version=args.neuron_version)
                elif args.install_type == 'update':
                    if self.df_package_properties.loc[self.df_package_properties['name'] == driver_package][
                        'pin_major'].values[0] == 'true':
                        version = '=' + self.get_package_version(category='driver', name=driver_package,
                                                                 neuron_version=args.neuron_version)
                str += f'sudo apt-get {install} {driver_package}{version}* -y'
                if args.install_type == 'update':
                    str += ' --allow-change-held-packages'
                str += '\n'

            elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os =='rockylinux9':
                yum_install = 'install' if args.install_type == 'install' else 'update'

                if args.install_type == 'install':

                    if args.neuron_version == None:
                        if self.df_package_properties.loc[self.df_package_properties['name'] == driver_package][
                            'pin_major'].values[0] == 'true':
                            version = '-' + self.get_major_version(driver_package, args.instance) + '.'
                    else:
                        version = '-' + self.get_package_version(category='driver', name=driver_package,
                                                                 neuron_version=args.neuron_version)
                elif args.install_type == 'update':
                    if self.df_package_properties.loc[self.df_package_properties['name'] == driver_package][
                        'pin_major'].values[0] == 'true':
                        version = '-' + self.get_major_version(driver_package, args.instance)

                str += f'sudo dnf {yum_install} {driver_package}{version}* -y\n'
        '''
        if args.ami == 'dlami-base':
            str += '--allow-change-held-packages'
        '''

        return str

    def install_neuron_runtime(self, args):
        """
        Neuron runtime install script will be generated based on instace, mode, AMI, and OS.
        instance: trn1
        mode: when develop or deploy
        AMI: when not dlami-base
        OS: for different command
        """
        str = ''

        # get runtime package names for release verion, instance

        runtime_packages = self.get_package_names(category='runtime', instance=args.instance,
                                                  neuron_version=args.neuron_version)
        # install neuron runtime on trn1
        if args.mode != 'compile':
            install = 'install' if args.install_type == 'install' else 'upgrade'
            if len(runtime_packages) != 0:
                if args.install_type == 'install':
                    str += '\n# Install Neuron Runtime \n'
                elif args.install_type == 'update':
                    str += '\n# Update Neuron Runtime\n'

                for runtime_package in runtime_packages:
                    # if args.ami != 'dlami-base':
                    if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                        str += (f'sudo apt-get {install} ' + runtime_package)
                        if args.neuron_version == None:
                            if self.df_package_properties.loc[self.df_package_properties['name'] == runtime_package][
                                'pin_major'].values[0] == 'true':
                                str += '=' + self.get_major_version(runtime_package, args.instance) + '.* -y'
                                if args.install_type == 'update':
                                    str += ' --allow-change-held-packages'
                                str += '\n'
                        elif (args.neuron_version != None) & (args.install_type == 'install'):
                            str += '=' + self.get_package_version(category='runtime', name=runtime_package,
                                                                  neuron_version=args.neuron_version) + '* -y\n'
                        else:
                            str += '\n'

                    elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
                        str += 'sudo dnf '
                        if args.install_type == 'install':
                            str += 'install '
                            str += runtime_package
                            if args.neuron_version == None:
                                if \
                                        self.df_package_properties.loc[
                                            self.df_package_properties['name'] == runtime_package][
                                            'pin_major'].values[0] == 'true':
                                    str += '-' + self.get_major_version(runtime_package, args.instance) + '.* -y\n'
                            else:
                                str += '-' + self.get_package_version(category='driver', name=runtime_package,
                                                                      neuron_version=args.neuron_version) + '* -y\n'
                        elif args.install_type == 'update':
                            str += 'update '
                            str += runtime_package
                            if self.df_package_properties.loc[self.df_package_properties['name'] == runtime_package][
                                'pin_major'].values[0] == 'true':
                                str += '-' + self.get_major_version(runtime_package, args.instance) + '.* -y\n'
        return str

    def install_efa_driver(self, args):
        str = ''
        # install EFA driver on trn1
        if args.instance == 'trn1' and args.mode == 'develop':
            str += '\n# Install EFA Driver (only required for multi-instance training)\n'
            str += 'curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \n'
            str += 'wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key \n'
            str += 'cat aws-efa-installer.key | gpg --fingerprint \n'
            str += 'wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig \n'
            str += 'tar -xvf aws-efa-installer-latest.tar.gz \n'
            str += 'cd aws-efa-installer && sudo bash efa_installer.sh --yes \n'
            str += 'cd \n'
            str += 'sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer \n'
        return str

    def install_neuron_system_tools(self, args):
        """
        Neuron tools will be installed in develop mode.
        """
        str = ''
        if args.mode == 'develop':
            # get runtime package names for release verion, instance
            install = 'install' if args.install_type == 'install' else 'upgrade'
            system_tool_packages = self.get_package_names(category='system-tools', instance=args.instance)
            if len(system_tool_packages) != 0:
                if args.install_type == 'install':
                    str += '\n# Install Neuron Tools \n'
                elif args.install_type == 'update':
                    str += '\n# Update Neuron Tools\n'

                for system_tool in system_tool_packages:
                    if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                        str += (f'sudo apt-get {install} ' + system_tool)
                        if args.neuron_version == None:
                            if self.df_package_properties.loc[self.df_package_properties['name'] == system_tool][
                                'pin_major'].values[0] == 'true':
                                str += '=' + self.get_major_version(system_tool, args.instance) + '.* -y'
                                if args.install_type == 'update':
                                    str += ' --allow-change-held-packages'
                                str += '\n'

                        elif (args.neuron_version != None) & (args.install_type == 'install'):
                            str += '=' + self.get_package_version(category='system-tools', name=system_tool,
                                                                  neuron_version=args.neuron_version) + '* -y\n'

                    elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
                        str += 'sudo dnf '
                        if args.install_type == 'install':
                            str += 'install '
                            str += system_tool
                            if args.neuron_version == None:
                                if self.df_package_properties.loc[self.df_package_properties['name'] == system_tool][
                                    'pin_major'].values[0] == 'true':
                                    str += '-' + self.get_major_version(system_tool, args.instance) + '.* -y\n'
                            else:
                                str += '-' + self.get_package_version(category='driver', name=system_tool,
                                                                      neuron_version=args.neuron_version) + '* -y\n'
                        elif args.install_type == 'update':
                            str += 'update '
                            str += system_tool
                            if self.df_package_properties.loc[self.df_package_properties['name'] == system_tool][
                                'pin_major'].values[0] == 'true':
                                str += '-' + self.get_major_version(system_tool, args.instance) + '.* -y\n'
        return str

    def install_extra_dependencies(self, args):
        """
        Any extra dependencies must be added in this function
        """
        str = ''
        if args.os == 'amazonlinux2023':
            str += '# Install External Dependency\n'
            str += 'sudo dnf '
            if args.mode == 'develop':
                str += 'install -y '
            elif args.install_type == 'update':
                str += 'update '
            str += 'libxcrypt-compat\n'
        return str

    def set_python_venv(self, args):
        # find the right python version that Neuron framework supports
        # (for fresh install) install the Python venv
        # (for fresh install and update) activate the venv
        str = ''

        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''

        os_default_python_version = \
            self.df_os_properties.loc[self.df_os_properties['os'] == args.os]['default_python_version'].values[0]
        packages_supporting_python_versions = self.get_pip_packages_supporting_python_versions(args)

        if os_default_python_version in packages_supporting_python_versions:
            target_python_version = os_default_python_version
        else:
            target_python_version = max(packages_supporting_python_versions)

        if args.install_type == 'install':
            # Install Python: if the default Python version of OS does not support Neuron packages, we install the supporting version
            if os_default_python_version not in packages_supporting_python_versions:
                str += '\n# Install Python \n'
                if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                    str += 'sudo add-apt-repository ppa:deadsnakes/ppa\n'
                    str += 'sudo apt-get install python' + target_python_version + '\n'
                elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023':
                    str += 'sudo dnf install -y amazon-linux-extras\n'
                    str += 'sudo dnf install python' + target_python_version + '\n'
                elif args.os == 'rockylinux9':
                    str += 'sudo dnf install python' + target_python_version + '\n'

            # Install Python venv
            """
            if os_default_python_version in packages_supporting_python_versions:
                str += '\n# Install Python venv \n'
                str +='python'+target_python_version+' -m venv '+args.framework+'_venv \n'
            else:
            """
            if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                str += '\n# Install Python venv \n'
                str += 'sudo apt-get install -y python' + target_python_version + '-venv g++ \n'
            elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023' or args.os == 'rockylinux9':
                str += '\n# Install Python venv \n'
                if args.os == 'amazonlinux2' or args.os == 'rockylinux9':
                    str += 'sudo dnf install -y python' + target_python_version + '-venv gcc-c++ \n'
                else:
                    str += 'sudo dnf install -y gcc-c++ \n'

            # when venv_install_type is parellel cluster, we need to change the directory
            if args.venv_install_type == 'parallel-cluster':
                if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                    str += '\ncd /home/ubuntu\n'
                elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023':
                    str += '\ncd /home/ec2-user\n'

                str += '. "/etc/parallelcluster/cfnconfig"\n'
                str += '\nif [[ $cfn_node_type == "HeadNode" ]]; then\n'

            # Create Python venv
            str += f'\n{indentation}# Create Python venv\n'
            str_venv_name = 'aws_neuron_venv_' + args.framework
            if args.instance == 'inf1':
                str_venv_name += '_inf1'

            str += f'{indentation}python{target_python_version} -m venv ' + str_venv_name + ' \n'

        return str

    def activate_python_venv(self, args):

        str = ''

        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''
        str_venv_name = ''
        str += f'\n{indentation}# Activate Python venv \n'

        if args.ami == 'dlami-framework':
            str_venv_name += '/opt/'

        str_venv_name += 'aws_neuron_venv_' + args.framework

        if args.instance == 'inf1':
            str_venv_name += '_inf1'

        str += f'{indentation}source ' + str_venv_name + '/bin/activate \n'

        # install python packages
        if (args.install_type == 'install' and args.ami != 'dlami-framework'):
            str += f'{indentation}python -m pip install -U pip \n'

        return str

    def set_pip_repository(self):
        str = ''

        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''

        str += f'\n{indentation}# Set pip repository pointing to the Neuron repository \n'
        str += f'{indentation}python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com\n'

        return str

    def install_aux(self, args):
        str = ''

        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''

        if args.instance == 'trn1':
            str += f'\n{indentation}# Install wget, awscli \n'
            str += f'{indentation}python -m pip install wget \n'
            str += f'{indentation}python -m pip install awscli \n'

        return str

    def install_neuron_compiler_and_framework(self, args):
        str = ''
        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''
        compiler_package = self.get_package_names(category='compiler', instance=args.instance)[0]
        framework_name = self.get_package_names(category=args.framework, instance=args.instance)[0]
        # if args.instance == 'inf1':
        #     return ''

        str = ''
        if args.mode != 'deploy':
            if args.install_type == 'install':
                str += f'\n{indentation}# Install Neuron Compiler and Framework\n'
            elif args.install_type == 'update':
                str += f'\n{indentation}# Update Neuron Compiler and Framework\n'

            str += f'{indentation}python -m pip install '
            if args.install_type == 'update':
                str += '--upgrade '


        str += compiler_package

        if args.neuron_version == None or args.install_type == 'update':
            if self.df_package_properties.loc[self.df_package_properties['name'] == compiler_package][
                'pin_major'].values[0] == 'true':
                str += '==' + self.get_major_version(compiler_package, args.instance) + '.* '
        else:
            str += '==' + self.get_package_version(category='compiler', name=compiler_package,
                                                   neuron_version=args.neuron_version) + ' '

        if args.neuron_version != None:  # prev install
            str += framework_name + '=='
            str += self.get_package_version(category=args.framework, name=framework_name,
                                            neuron_version=args.neuron_version,
                                            framework_version=args.framework_version)
        else:  # fresh install
            if args.framework == 'pytorch':
                str += framework_name
                if args.framework_version == "1.13.1":
                    str += '=='
                    str += "1.13.*"
                elif args.framework_version == "2.1.2":
                    str += '=='
                    str += "2.1.*"
                elif args.framework_version == "2.5.1":
                    str += '=='
                    str += "2.5.*"
                elif args.framework_version == "2.6.0":
                    str += '=='
                    str += "2.6.*"
                elif args.framework_version == "2.7.0":
                    str += '=='
                    str += "2.7.*"
                elif args.framework_version == "2.8.0":
                    str += '=='
                    str += "2.8.*"
                str += ' torchvision\n'
            else:
                str += framework_name

        if args.instance == 'inf1':

            install = 'Install' if args.install_type == 'install' else 'Update'
            upgrade = '--upgrade ' if args.install_type == 'update' else ''

            if args.neuron_version != None:  # in case of previous neuron version
                version = '==' + self.get_package_version(category=args.framework, neuron_version=args.neuron_version,
                                                          framework_version=args.framework_version)
            else:  # in case of latest neuron version (fresh install)
                if args.framework_version.startswith(
                        self.get_main_framework_version(instance=args.instance, framework=args.framework,
                                                        neuron_version=args.neuron_version)) == False:
                    version = '==' + args.framework_version + '.*'
                else:
                    version = ''

            if args.framework == 'pytorch':

                pytorch_aux = ' neuron-cc[tensorflow] "protobuf"' if args.mode != 'deploy' else ''

                str = f'\n# {install} PyTorch Neuron\n'
                str += f'python -m pip install {upgrade}torch-neuron{version}{pytorch_aux} torchvision\n'

            elif args.framework == 'tensorflow':

                if args.neuron_version != None:  # in case of previous neuron version

                    ms_version = '=' + self.get_package_version(category='model-server',
                                                                neuron_version=args.neuron_version,
                                                                framework_version=args.framework_version)
                else:  # in case of latest neuron version (fresh install)
                    if args.framework_version != self.get_main_framework_version(instance=args.instance,
                                                                                 framework=args.framework,
                                                                                 neuron_version=args.neuron_version):
                        ms_version = '=' + self.get_package_version(category='model-server',
                                                                    neuron_version=args.neuron_version,
                                                                    framework_version=args.framework_version)
                    else:
                        ms_version = ''

                str = f'\n# {install} TensorFlow Neuron\n'
                str += f'python -m pip install {upgrade}tensorflow-neuron[cc]{version} "protobuf"\n'

                str += f'\n# {install} Neuron TensorBoard\n'
                str += f'python -m pip install {upgrade}tensorboard-plugin-neuron\n'

                if args.mode != 'compile':
                    str += f'\n# Optional: {install} Tensorflow Neuron model server\n'
                    if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                        str += f'sudo apt-get install tensorflow-model-server-neuronx{ms_version} -y\n'
                    elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023':
                        str += f'sudo dnf install tensorflow-model-server-neuronx{ms_version} -y\n'

            elif args.framework == 'mxnet':

                mxnet_framework = ''

                neuron_cc_version = ''
                if args.framework_version == '1.8.0':
                    mxnet_framework = 'mx_neuron'
                elif args.framework_version == '1.5.1':
                    mxnet_framework = 'mxnet_neuron'
                    neuron_cc_version='==1.15.0'

                str = f'\n# {install} MXNet Neuron\n'
                str += 'wget https://aws-mx-pypi.s3.us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl\n'
                str += 'pip install aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl\n'
                str += f'python -m pip install {upgrade}{mxnet_framework}{version} neuron-cc{neuron_cc_version}\n'

        if args.venv_install_type == 'parallel-cluster':
            if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                str += f'\n\n{indentation}chown ubuntu:ubuntu -R {args.framework}_venv\n'
            elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023':
                str += f'\n\n{indentation}chown ec2-user:ec2-user -R {args.framework}_venv\n'

            str += 'fi'

        return str

    def install_neuron_compiler(self, args):
        '''
        Neuron compiler will be installed in develop or compile mode based on the instance.
        '''
        str = ''

        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''

        compiler_package = self.get_package_names(category='compiler', instance=args.instance)[0]

        if args.instance == 'inf1':
            return ''

        str = ''
        if args.mode != 'deploy':
            if args.install_type == 'install':
                str += f'\n{indentation}# Install Neuron Compiler\n'
            elif args.install_type == 'update':
                str += f'\n{indentation}# Update Neuron Compiler\n'

            str += f'{indentation}python -m pip install '
            if args.install_type == 'update':
                str += '--upgrade '

            str += compiler_package

            if args.neuron_version == None:
                if self.df_package_properties.loc[self.df_package_properties['name'] == compiler_package][
                    'pin_major'].values[0] == 'true':
                    str += '==' + self.get_major_version(compiler_package, args.instance) + '.* \n'
                else:
                    str += '\n'
            else:
                str += '==' + self.get_package_version(category='compiler', name=compiler_package,
                                                       neuron_version=args.neuron_version) + '\n'

        return str

    def install_neuron_framework(self, args):
        '''
        Neuron framework is installed based on:
        instance
        framework
        framework-version
        '''
        str = ''
        indentation = '\t' if args.venv_install_type == 'parallel-cluster' else ''

        framework_name = self.get_package_names(category=args.framework, instance=args.instance)[0]

        if args.install_type == 'install':
            str += f'\n{indentation}# Install Neuron Framework\n'
        elif args.install_type == 'update':
            str += f'\n{indentation}# Update Neuron Framework\n'

        str += f'{indentation}python -m pip install '
        if args.install_type == 'update':
            str += '--upgrade '

        if args.neuron_version != None:  # prev install
            str += framework_name + '=='
            str += self.get_package_version(category=args.framework, name=framework_name,
                                            neuron_version=args.neuron_version,
                                            framework_version=args.framework_version)
        else:  # fresh install
            str += framework_name

        if args.framework == 'pytorch':
            str += ' torchvision\n'

        if args.instance == 'inf1':

            install = 'Install' if args.install_type == 'install' else 'Update'
            upgrade = '--upgrade ' if args.install_type == 'update' else ''

            if args.neuron_version != None:  # in case of previous neuron version
                version = '==' + self.get_package_version(category=args.framework, neuron_version=args.neuron_version,
                                                          framework_version=args.framework_version)
            else:  # in case of latest neuron version (fresh install)
                if args.framework_version.startswith(
                        self.get_main_framework_version(instance=args.instance, framework=args.framework,
                                                        neuron_version=args.neuron_version)) == False:
                    version = '==' + args.framework_version + '.*'
                else:
                    version = ''

            if args.framework == 'pytorch':

                pytorch_aux = ' neuron-cc[tensorflow] "protobuf"' if args.mode != 'deploy' else ''

                str = f'\n# {install} PyTorch Neuron\n'
                str += f'python -m pip install {upgrade}torch-neuron{version}{pytorch_aux} torchvision\n'


            elif args.framework == 'tensorflow':

                if args.neuron_version != None:  # in case of previous neuron version

                    ms_version = '=' + self.get_package_version(category='model-server',
                                                                neuron_version=args.neuron_version,
                                                                framework_version=args.framework_version)
                else:  # in case of latest neuron version (fresh install)
                    if args.framework_version != self.get_main_framework_version(instance=args.instance,
                                                                                 framework=args.framework,
                                                                                 neuron_version=args.neuron_version):
                        ms_version = '=' + self.get_package_version(category='model-server',
                                                                    neuron_version=args.neuron_version,
                                                                    framework_version=args.framework_version)
                    else:
                        ms_version = ''

                str = f'\n# {install} TensorFlow Neuron\n'
                str += f'python -m pip install {upgrade}tensorflow-neuron[cc]{version} "protobuf"\n'

                if args.mode != 'compile':
                    str += f'\n# Optional: {install} Tensorflow Neuron model server\n'
                    if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                        str += f'sudo apt-get install tensorflow-model-server-neuronx{ms_version} -y\n'
                    elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023':
                        str += f'sudo dnf install tensorflow-model-server-neuronx{ms_version} -y\n'

            elif args.framework == 'mxnet':

                mxnet_framework = ''

                if args.framework_version == '1.8.0':
                    mxnet_framework = 'mx_neuron'
                elif args.framework_version == '1.5.1':
                    mxnet_framework = 'mxnet_neuron'

                str = f'\n# {install} MXNet Neuron\n'
                str += 'wget https://aws-mx-pypi.s3.us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl\n'
                str += 'pip install aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl\n'
                str += f'python -m pip install {upgrade}{mxnet_framework}{version} neuron-cc\n'

        if args.venv_install_type == 'parallel-cluster':
            if args.os == 'ubuntu18' or args.os == 'ubuntu20' or args.os == 'ubuntu22':
                str += f'\n\n{indentation}chown ubuntu:ubuntu -R {args.framework}_venv\n'
            elif args.os == 'amazonlinux2' or args.os == 'amazonlinux2023':
                str += f'\n\n{indentation}chown ec2-user:ec2-user -R {args.framework}_venv\n'

            str += 'fi'

        return str

    def get_latest_neuron_version_per_instance(self, instance):
        return self.df_latest_release.loc[self.df_latest_release['instance'] == instance]['version'].values[0]

    def get_package_names(self, category, instance, neuron_version=None):

        if neuron_version == None:
            neuron_version = self.get_latest_neuron_version_per_instance(instance)

        df_instance = self.df_release_packages[
            self.df_release_packages['supported_instances'].map(lambda x: instance in x)]

        return \
            df_instance.loc[(df_instance['category'] == category) & (df_instance['neuron_version'] == neuron_version)][
                'name'].tolist()

    def get_package_version(self, category, neuron_version, name=None, framework_version=None):
        if neuron_version == None:
            neuron_version = self.get_latest_neuron_version_per_instance(args.instance)

        if name != None:
            df_package = self.df_release_packages.loc[(self.df_release_packages['neuron_version'] == neuron_version) & (
                    self.df_release_packages['name'] == name)]
        else:
            df_package = self.df_release_packages.loc[self.df_release_packages['neuron_version'] == neuron_version]

        if (category == 'pytorch') or (category == 'tensorflow') or (category == 'mxnet') or (
                category == 'model-server'):
            df_package = df_package.loc[df_package['category'] == category]
            fv = self.extract_major_minor_version(Version(framework_version))
            df_package = df_package.loc[df_package['version'].map(lambda x: x.startswith(fv))]
        return df_package['version'].values[0]

    def get_main_framework_version(self, instance, framework, neuron_version):

        if neuron_version == None:
            neuron_version = self.get_latest_neuron_version_per_instance(instance)

        df_instance = self.df_release_packages[
            self.df_release_packages['supported_instances'].map(lambda x: instance in x)]

        df_version = df_instance.loc[
            (df_instance['category'] == framework) & (df_instance['neuron_version'] == neuron_version)].copy()

        df_version['version'] = df_version['version'].map(lambda x: Version(x))

        main_version = sorted(df_version['version'], reverse=True)[0]

        return str(main_version.major) + '.' + str(main_version.minor)

    def list_packages(self, args):

        str = ''

        if args.neuron_version == None:
            neuron_version = self.get_latest_neuron_version_per_instance(args.instance)
        else:
            neuron_version = args.neuron_version

        if (args.list == 'packages'):  # list packages by neuron version

            df_instance = self.df_release_packages[
                self.df_release_packages['supported_instances'].map(lambda x: args.instance in x)]

            df_version = df_instance.loc[
                (df_instance['neuron_version'] == neuron_version) & (df_instance['category'] != 'efa')].copy()

            str += '\nList of packages in Neuron ' + neuron_version + ':\n\n'
            str += '{0:35} {1:50}\n'.format("Component", "Package")

            for index, row in df_version.iterrows():
                if row['category'] == 'libnrt':
                    str += f"{row['component']:<35} {row['name'] + ' (Version ' + row['version']})\n"
                else:
                    str += f"{row['component']:<35} {row['name'] + '-' + row['version']} \n"

            df_version['package'] = (df_version['name'] + '-' + df_version['version'])

        return str

    def list_pyversions(self, args):

        str = ''

        if args.neuron_version == None:
            neuron_version = self.get_latest_neuron_version_per_instance(args.instance)
        else:
            neuron_version = args.neuron_version

        if (args.list == 'pyversions'):  # list packages by neuron version

            df_instance = self.df_release_packages[
                self.df_release_packages['supported_instances'].map(lambda x: args.instance in x)]

            df_version = df_instance.loc[
                (df_instance['neuron_version'] == neuron_version) & (df_instance['category'] != 'efa')].copy()

            str += '\nList of packages in Neuron ' + neuron_version + ':\n\n'
            str += '{0:35} {1:50}\n'.format("Package", "           Supported Python Versions")

            for index, row in df_version.iterrows():
                python_version_str = ''
                for i, pversion in enumerate(row['supported_python_versions']):
                    if i != len(row['supported_python_versions'])-1:
                        python_version_str += pversion + ", "
                    else:
                        python_version_str += pversion
                if len(row['supported_python_versions']) != 0:
                    if row['category'] == 'libnrt':
                        str += f"{row['name'] + ' Version ' + row['version']:<50}{python_version_str} \n"
                    else:
                        str += f"{row['name'] + '-' + row['version']:<50}{python_version_str} \n"

                df_version['package'] = (df_version['name'] + '-' + df_version['version'])

        return str



################
# Sanity Checks
################
def cli_validate(args):
    # case of parallel-cluster, it should not be inf1
    if (args.venv_install_type == 'parallel-cluster') & (args.instance == 'inf1'):
        print(__name__, ": error: ", "parallel-cluster scripts is not compatible with inf1")
        exit(-1)


########################################
# parse_arguments
########################################

def cli_parse_arguments():
    __name__ = 'n2-helper.py'
    parser = argparse.ArgumentParser(prog=__name__
                                     ,
                                     usage='\npython3 %(prog)s --list={packages} [--neuron-version=X.Y.Z] [--instance=INSTANCE]\n'
                                           + 'python3 %(prog)s --list={pyversions} [--neuron-version=X.Y.Z] [--instance=INSTANCE]\n'
                                           + 'python3 %(prog)s --install-type={install,update}\n'
                                           + 'python3 %(prog)s --instance={inf1,trn1,inf2,trn2}\n'
                                           + 'python3 %(prog)s --os={ubuntu18,ubuntu20,ubuntu22,amazonlinux2,amazonlinux2023,rockylinux9}\n'
                                           + 'python3 %(prog)s --ami={non-dlami,dlami-base,dlami-conda,dlami-framework,dlami-neuron}\n'
                                           + 'python3 %(prog)s --framework={pytorch,tensorflow,mxnet}\n'
                                           + 'python3 %(prog)s --framework-version=[X.Y.Z] [options]\n'
                                           + 'python3 %(prog)s --mode={develop,compile,deploy} [options]\n'
                                           + 'python3 %(prog)s --category={framework,driver,runtime,compiler,tools,all,driver_runtime_tools,compiler_framework,efa, transformers-neuronx}\n'
                                           + 'options= [--file=FILE]\n'
                                     , description='Installer helper for Neuron SDK')

    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--neuron-version", metavar='X.Y.Z')
    group.add_argument("--list", choices=['neuron_versions', 'pyversions','packages', 'components', 'frameworks'])
    group.add_argument("--install-type", choices=['install', 'update'])
    parser.add_argument("--instance", choices=['inf1', 'trn1', 'inf2', 'trn2'])
    parser.add_argument("--os", choices=['ubuntu18', 'ubuntu20', 'ubuntu22', 'amazonlinux2', 'amazonlinux2023', 'rockylinux9'], )
    parser.add_argument("--ami", choices=['non-dlami', 'dlami-base', 'dlami-conda', 'dlami-framework', 'dlami-neuron'],
                        default='non-dlami', help='default=non-dlami')
    parser.add_argument("--mode", choices=['develop', 'compile', 'deploy', 'initialize'], default='develop')
    parser.add_argument("--category",
                        choices=['framework', 'driver', 'runtime', 'compiler', 'tools', 'all', 'driver_runtime_tools',
                                 'compiler_framework', 'efa', 'transformers-neuronx'])
    parser.add_argument("--framework", choices=['pytorch', 'tensorflow', 'mxnet'])
    parser.add_argument("--framework-version", metavar='X.Y.Z')
    parser.add_argument("--venv-install-type", choices=['single-node', 'parallel-cluster'], default='single-node')
    parser.add_argument("--file", default='n2-manifest.json', help='default=n2-manifest.json')

    return parser.parse_args()


if __name__ == '__main__':
    setup_cmd = ''
    args = cli_parse_arguments()

    # arguments sanity check
    cli_validate(args)

    # parse the manifest file
    n2_manifest = manifest(manifest_file=args.file)
    n2_manifest.parse_manifest()

    # framework version sanity check
    # generate install script
    if (args.list == 'packages'):
        print(n2_manifest.list_packages(args))
    elif (args.list == 'pyversions'):
        print(n2_manifest.list_pyversions(args))
    else:
        print(n2_manifest.generate_script(args))
