#!/usr/bin/env python3
# coding=utf-8


""" Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

    SPDX-License-Identifier: MIT-0

    Program to gather information from a system
"""
import sys
import os
import argparse
import shutil
import subprocess
import re

ACTUAL_CMD = os.path.realpath(sys.argv[0])

USAGE_MSG = """
    Usage: {} [options]
    This program is used to gather information from this system for analysis
    and debugging
    """.format(ACTUAL_CMD)

EXCLUDE_FILES_BY_NAME = "weight files, model, NEFF (Neuron Executable File Format)"

HELP_CC_FILES = """ Location of the neuron-cc generated files """
DEFAULT_CCFILES_LOCATION = "~/bin"

SYSLOG_SEARCH_PATTERNS = r"nrtd|neuron|kernel:"

EXTERNAL_CMDS = ["lscpu", "lshw",
                 "lspci | grep -i Amazon",
                 "neuron-cc --version",
                 "neuron-ls",
                 "top -b -n 1",
                 "uname -a", "uptime",
                 ]

PROC_FILES = ["/proc/cmdline",
              "/proc/cpuinfo",
              "/proc/filesystems",
              "/proc/interrupts",
              "/proc/iomem",
              "/proc/loadavg",
              "/proc/meminfo",
              "/proc/modules",
              "/proc/mtrr",
              "/proc/version",
              ]

HELP_ADDITIONAL_FILE_OR_DIR = """ Additional file or directory that the user wants to provide in
    the archive. The user can sanitize this file or directory before sharing """

INCLUDE_MSG = """
    By default, only the lines containing (grep) patterns like '{}' from the syslog are copied.
    Other lines are excluded. Using this option allows the timestamp section of other lines
    to be included. The rest of the contents of the line itself are elided. Providing the
    timestamp section may provide time continuity while viewing the copied syslog file
    """.format(SYSLOG_SEARCH_PATTERNS)

HELP_RT_FILES = """ Location of the neuron runtime generated files """
MISCINFO_FILE = 'miscinfo.txt'

HELP_VERBOSE = """ Verbose mode displays commands executed and any additional information
                   which may be useful in debugging the tool itself
               """

INCLUDE_EXTNS = ('.pb')

HELP_INCLUDE_EXTN_FILES = """ Include files with these extensions from the compiler work
    directory in the archive:
    {}
    """.format(INCLUDE_EXTNS)

HELP_STDOUT = """ The file where the stdout of the compiler run was saved """

HELP_OUTDIR_MSG = """
    The output directory where all the files and other information will be stored.
    The output will be stored as an archive as well as the actual directory where all the
    contents are copied. This will allow a simple  audit of the files, if necessary.
    *** N O T E ***: Make sure that this directory has enough space to hold the files
    and resulting archive
    """

USERCMDFILE = "how-the-user-executed-the-script-{}.txt".format(os.path.basename(ACTUAL_CMD))

NEURONDUMPPROGRAM = "/opt/aws/neuron/bin/neuron-dump.py"
NEURONDUMPFILE = os.path.splitext(os.path.basename(NEURONDUMPPROGRAM))[0]

NEURON_ERRMSG = "Error: File {} doesn't exist, aws-neuron-tool package isn't installed?".format(
        NEURONDUMPPROGRAM)

NEURON_INFO_TARBALL = "{}".format(os.path.splitext(os.path.basename(ACTUAL_CMD))[0])
NEURONTMPDIR = NEURON_INFO_TARBALL

ARCHIVE_MSG = "\n\n\t******\n\tArchive created at:\n\t\t{}\n\tFrom directory:\n\t\t{}\n\t******\n\n"

NOT_IMPLEMENTED_MSG = ", nothing to see here, folks (not implemented as yet)"

# these are the only compiler-generated files that are included by default
COMPILER_FILES = ['graph_def.neuron-cc.log', 'all_metrics.csv', 'hh-tr-operand-tensortensor.json']

COMPILER_FILES_USER_OPT_IN = ['exp_and_others.json', 'graph_def.neff', 'graph_def.pb',
                              'hh-spilled.json', 'hh-tr-accDN2virtDN.json',
                              'hh-tr-external-move.json', 'hh-tr-internal-move.json',
                              'hh-tr-removeDN.json', 'hh-transforms.json', 'wavegraph.json',
                              'hh.json', 'pass03_scheduling.json',
                              'relay_graph_opt_pre_color.txt', 'relay_graph_post_opt_kelp.txt',
                              'relay_graph_post_opt_unit_level.txt', 'relay_graph_pre_opt.txt',
                              'saved_model.pb', 'sch.json', 'sch_tmp.json',
                              'schedule_trace.json',
                              'wavegraph-bin.json']

MODEL_DATA_MSG = """
    By using this option, the entire compiler work directory's contents will be
    included (excluding the {} files, unless an additional option is used). This would
    include model information, etc.
    The files that are included, by default, are these: {}

    """.format(INCLUDE_EXTNS, ", ".join(COMPILER_FILES))

MODEL_DATA_MSG_INFO = """
\t**************************
\tBased on your command line option, we're also packaging these files:

\t\t{}

\tAnd this directory: {}

\t**************************
"""

def get_os_version():

    ''' function to obtain the Linux version
        Args:

        Output:

        Returns:
            string with value 'Ubuntu' or 'RedHat'
    '''

    try:
        with open("/proc/version") as fdin:
            data = fdin.read()
            if data.find('Ubuntu') == -1:
                osver = 'RedHat'
            else:
                osver = 'Ubuntu'
    except FileNotFoundError:
        osver = 'Ubuntu'

    return osver


def get_files(*, basedir, matchfiles, verbose):
    ''' function to get the files based on a base directory and file extension

        Args:
            basedir     : base directory where files reside
            matchfiles  : set of files to match
            verbose : flag to indicate if verbose messages need to be displayed

        Output:

        Returns:
            list of files found

    '''

    myfiles = list()
    for dpath, _, files in os.walk(basedir):
        for mfile in files:
            if mfile in matchfiles:
                mfile = os.path.realpath(os.path.join(dpath, mfile))
                if os.path.isfile(mfile):
                    myfiles.append(mfile)
                else:
                    if verbose:
                        print("Warning: {} is not a file".format(mfile))

    return myfiles


def dump_compiler_info(*, outdir, location, allowmodel=False, addfldir=None, verbose=False):
    ''' function to gather the following information:
            Framework:
                - TensorFlow
                - MXNet
                - PyTorch
            Compiler:
        Args:
            outdir      : output directory
            location    : location of compiler-generated files
            allowmodel  : if True, allow gathering of additional files
            verbose : flag to indicate if verbose messages need to be displayed

        Output: compiler-generated files copied to outdir

        Returns:
    '''

    if location is not None:
        if allowmodel:  # copy the entire directory
            try:
                shutil.copytree(location, os.path.join(outdir, os.path.basename(location)),
                                ignore_dangling_symlinks=True)
            except shutil.Error:
                pass
        else:
            fileset = set(COMPILER_FILES)
            l1data = get_files(basedir=location, matchfiles=fileset, verbose=verbose)
            copy_files(outdir=outdir, basedir=location, filelist=l1data, verbose=verbose)

        if addfldir is not None:
            if os.path.isfile(addfldir):
                shutil.copy(addfldir, outdir)
            else:  # directory copy
                try:
                    shutil.copytree(addfldir, os.path.join(outdir, os.path.basename(addfldir)),
                                    ignore_dangling_symlinks=True)
                except shutil.Error:
                    pass

    # print("Function: ", sys._getframe().f_code.co_name,  # pylint: disable=W0212
    #       NOT_IMPLEMENTED_MSG)


def copy_stdout(*, outdir, stdout, verbose):
    ''' function to copy the stdout file to the destination location

        Args:
            outdir  : destination location (output directory)
            stdout  : file containing the output of running neuron-cc
            verbose : flag to indicate if verbose messages need to be displayed

        Output:

        Returns:
    '''

    if verbose:
        print("Copying {} to {}".format(stdout, outdir))

    shutil.copy(stdout, outdir)


def copy_syslog(*, outdir, include_flag=False, verbose):
    '''
        function to copy contents of the syslog to the output directory

        Args:
            outdir          : output directory location where the syslog's contents
                              are to be copied
            include_flag    : if True, include lines that do not match
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            copy of syslog's contents with just "Neuron-specific" lines

        Returns:
    '''

    # syslog looks like this:
    # 2019-11-21T19:32:50.347183+00:00 ink neuron-rtd[17977]: nrtd[17977]: <SNIP>
    # The first regex (regex1) is used to match lines that we want to see in our copy

    regex1 = re.compile(r'^(\S+)\s.*?({})'.format(SYSLOG_SEARCH_PATTERNS))
    regex2 = re.compile(r'^(\S+)\s')

    osver = get_os_version()
    if osver == 'Ubuntu':
        syslog = '/var/log/syslog'
    else:
        syslog = '/var/log/messages'

    try:
        with open(syslog) as fdin,\
            open(os.path.join(outdir, 'copy-of-syslog'), 'w') as fdout:
            for line in fdin:
                match = regex1.search(line)
                if match is not None:
                    fdout.write(line)
                else:
                    if include_flag:
                        match = regex2.match(line)
                        if match is not None:
                            # exclude the rest of the line
                            fdout.write(match.group(1) + ' XXX contents elided XXX\n')
                        else:
                            print("Error in parsing this line: {}".format(line))
    except FileNotFoundError:
        print("Error, /var/log/syslog not found")


def dump_rt_info(*, location, verbose):
    ''' function to dump the following information:
            - runtime
            - Framework (??)
        Args:
            location: location of runtime files
            verbose : flag to indicate if verbose messages need to be displayed
        Returns:
            list of info
    '''

    # l1data = get_files(basedir=location, file_extn=('.sh'))
    print("Function: ", sys._getframe().f_code.co_name,  # pylint: disable=W0212
          NOT_IMPLEMENTED_MSG)


def allow_capture_of_files():
    '''
        function to allow the capture of files from the customer's environment
        This is OFF by default and has to be explicitly enabled by the command-line
        option by the user

        Args:

        Output:

        Returns:

    '''

    print("Function: ", sys._getframe().f_code.co_name,  # pylint: disable=W0212
          NOT_IMPLEMENTED_MSG)


def add_additional_filters(filterfile):
    '''
        function to apply additional filters to files that are being captured

        Args:
            filterfile  : text file with patterns (regexs), one per line, to use as filters


        Output:

        Returns:

    '''

    print("Function: ", sys._getframe().f_code.co_name,  # pylint: disable=W0212
          NOT_IMPLEMENTED_MSG)


def dump_miscinfo(*, outdir, verbose):
    ''' function to dump miscellaneous information, including:
            - system info (uname -a)
            - package info (??? list of packages installed)
            - neuron-ls
            - neuron-top

        Args:
            outdir  : output directory
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            Creates various reports in the outdir location

        Returns:

    '''

    osver = get_os_version()
    if osver == 'Ubuntu':
        pkgcmds = ["apt list | egrep '^aws'",
                   "pip list | egrep '^neuron|^numpy|^tensor|^scipy'"]
    else:
        pkgcmds = ["rpm -qa | egrep '^aws|^neuron|^numpy|^tensor|^scipy'"]

    cmds = EXTERNAL_CMDS + pkgcmds

    for cmd in cmds:
        cmdname = cmd.split(' ')[0]  # get just the command name for creating the file
        cmdfile = os.path.join(outdir, "report-{}.txt".format(cmdname))

        with open(cmdfile, "w") as fdout:

            if verbose:
                print("Running cmd: {} and capturing output in file: {}".format(cmd, cmdfile))

            try:
                res = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, universal_newlines=True,
                                       shell=True)
                stdout, stderr = res.communicate()
                if stderr is not None:
                    fdout.write("Error in executing cmd: {}\nError: {}\n".format(cmd, str(stderr)))
                else:
                    fdout.write("Output from executing cmd: {}\n\n{}\n".format(cmd, str(stdout)))
            except (OSError, ValueError) as err:
                fdout.write("Error in executing cmd: {}\nError: {}\n".format(cmd, err))


def dump_proc_info(*, outdir, verbose):
    '''
        function to dump information related to "/proc"

        Args:
            outdir  : output directory
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            Creates various reports in the outdir location

        Returns:

    '''

    for procfile in PROC_FILES:
        fname = procfile.split('/')  # use the 2nd and 3rd items from this (canonical form)
        pfile = os.path.join(outdir, "report-{}-{}.txt".format(fname[1], fname[2]))
        if verbose:
            print("Copying contents of: {} to: {}".format(procfile, pfile))

        try:
            with open(pfile, "w") as fdout, open(procfile) as fdin:
                fdout.write("Contents of {}\n\n".format(procfile))
                fdout.write(fdin.read())
        except FileNotFoundError:
            print("Error: file {} not found\n".format(procfile))


def sanity_check(options):
    '''
        function to check if command-line arguments are valid

        Args:
            options : options from argparse parser

        Output:

        Returns:
            0 : success
            1 : failure
    '''

    # the script has to be run as root or "sudo"
    if os.getuid() != 0:
        print("*** Rerun this script as user 'root' or as sudo **\n\n")
        return 1

    outdir = options.outdir

    retval = 0
    if os.path.isfile(outdir) or os.path.isdir(outdir):
        print("Error: {} already exists, please provide a non-existing directory".format(outdir))
        retval = 1

    if not os.path.isfile(options.stdout):
        print("Error: {} doesn't exist, please provide an existing file".format(options.stdout))
        retval = 1

    if options.addfldir is not None:
        if not os.path.isfile(options.addfldir) and not os.path.isdir(options.addfldir):
            print("Error: {} isn't a file nor a directory".format(options.addfldir))
            retval = 1

    for mydir in [options.ccdir, options.rtdir]:
        if mydir is not None and not os.path.isdir(mydir):
            print("Error: {} is not a directory, please provide a directory".format(mydir))
            retval = 1

    if options.allowmodel and options.ccdir is None:
        print("Error: you need to specify a compiler work directory along with the 'm' option")
        retval = 1
    return retval


def copy_files(*, outdir, basedir, filelist, verbose):
    '''
        function to copy files from the original source area
        into the destination. This is also the place for any
        massaging or eliding of file contents

        Args:
            outdir  : destination location
            basedir : base directory from where the files are to be copied
            filelist: list of files to be copied
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            Copy of files (possibly altered) from the source

        Returns:

    '''

    for thisfile in filelist:
        myfile = '.' + thisfile[len(basedir):]
        mydir = os.path.dirname(os.path.join(outdir, myfile))
        if not os.path.isdir(mydir):
            os.makedirs(mydir)
        shutil.copy(thisfile, mydir, follow_symlinks=True)


def write_miscinfo(*, outdir, data):
    '''
        function to write out the contents of the miscellaneous commands

        Args:
            outdir  : destination location
            data    : list of strings to be stored in a file

        Output:
            MISCINFO_FILE created with the contents of the output of the various
            commands
    '''

    flname = os.path.join(outdir, MISCINFO_FILE)

    with open(flname, "w") as fdout:
        fdout.write("\n".join(data))


def run_neuron_dump(outdir, verbose):
    '''
        function to call the existing neuron-dump.py tool

        Args:
            outdir  : destination location
            verbose : flag to indicate if verbose messages need to be displayed

        Output:
            tarball created by this tool

        Returns:

    '''

    if not os.path.isfile(NEURONDUMPPROGRAM):
        print(NEURON_ERRMSG)
        return

    cmd = "{} -o {}".format(NEURONDUMPPROGRAM, os.path.join(outdir, NEURONDUMPFILE))

    if verbose:
        print("Executing command: {}".format(cmd))

    try:
        res = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True,
                               shell=True)
        stdout, stderr = res.communicate()
        if stderr is not None:
            print("Error in executing cmd: {}\nError: {}\n".format(cmd, str(stderr)))
    except (OSError, ValueError) as err:
        print("Error in executing cmd: {}\nError: {}\n".format(cmd, err))

    if verbose:
        print("Output of cmd: {}\n{}".format(cmd, stdout))


def package_tarball(*, outdir, allowmodel, ccdir, verbose):
    '''
        function to package everything into a tarball

        Args:
            outdir      : output directory
            allowmodel  : flag to indicate whether the user has allowed
                          gathering of model data

        Output:
            A tar ball created in directory one level above outdir
            this would be the directory provided by the user

        Returns:
    '''

    mytarball = os.path.join(os.path.split(outdir)[0], NEURON_INFO_TARBALL)

    if verbose:
        print("Creating archive: {}".format(mytarball))

    archivefile = shutil.make_archive(mytarball, 'gztar', outdir)
    print(ARCHIVE_MSG.format(archivefile, outdir))

    if allowmodel:
        print(MODEL_DATA_MSG_INFO.format("\n\t\t".join(COMPILER_FILES),
                                         ccdir))


def add_cmdline_args():
    '''
        function to add the command line arguments and options

        Args:

        Output:

        Returns:
            parser for cmd line

    '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=USAGE_MSG)

    parser.add_argument('--additionalfileordir',
                        dest='addfldir',
                        help=HELP_ADDITIONAL_FILE_OR_DIR,
                        default=None)

    parser.add_argument('-c', '--compileroutdir',
                        dest='ccdir',
                        help=HELP_CC_FILES,
                        default=None)

    parser.add_argument('-i', '--include',
                        dest='includemismatch',
                        help=INCLUDE_MSG,
                        action='store_true',
                        default=False)

    parser.add_argument('-f', '--filter',
                        dest='filterfile',
                        default=None)

    parser.add_argument('-m', "--modeldata",  # data related to model, etc. will be gathered
                        dest='allowmodel',
                        action='store_true',
                        help=MODEL_DATA_MSG,
                        default=False)

    parser.add_argument('-o', '--out',
                        dest='outdir',
                        help=HELP_OUTDIR_MSG,
                        required=True)

    parser.add_argument('-r', '--runtimeoutdir',
                        dest='rtdir',
                        help=HELP_RT_FILES,
                        default=None)

    parser.add_argument('-s', '--stdout',
                        dest='stdout',
                        help=HELP_STDOUT,
                        required=True)

    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        help=HELP_VERBOSE,
                        action='store_true',
                        default=False)

    return parser


def main():
    """ main function
        creates command-line option parser, sanity checks, and then executes code
        based on command-line options
    """

    parser = add_cmdline_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    options = parser.parse_args()
    # append the directory where we'll create files to what the user provides
    options.outdir = os.path.realpath(os.path.join(options.outdir, NEURONTMPDIR))

    if options.ccdir is not None:
        options.ccdir = os.path.realpath(options.ccdir)

    if options.addfldir is not None:
        options.addfldir = os.path.realpath(options.addfldir)

    if options.rtdir is not None:
        options.rtdir = os.path.realpath(options.rtdir)

    options.stdout = os.path.realpath(options.stdout)

    if sanity_check(options):
        parser.print_help()
        sys.exit(1)

    # create the base directory
    try:
        os.makedirs(options.outdir)
    except FileNotFoundError:
        print("Error in creating directory {}".format(options.outdir))
        sys.exit(1)

    # if options.allow:
    #     allow_capture_of_files()

    if options.filterfile is not None:
        add_additional_filters(os.path.realpath(options.filterfile))

    # record the command as executed by the user
    with open(os.path.join(options.outdir, USERCMDFILE), "w") as fdout:
        fdout.write("Command executed as: {}\n".format(" ".join(sys.argv)))

    dump_compiler_info(outdir=options.outdir, location=options.ccdir,
                       allowmodel=options.allowmodel,
                       addfldir=options.addfldir,
                       verbose=options.verbose)

    # Not being used now. neuron-dump.py would do this
    # dump_rt_info(location=options.rtdir, verbose=options.verbose)

    dump_miscinfo(outdir=options.outdir, verbose=options.verbose)
    dump_proc_info(outdir=options.outdir, verbose=options.verbose)

    copy_stdout(outdir=options.outdir, stdout=options.stdout, verbose=options.verbose)
    copy_syslog(outdir=options.outdir, include_flag=options.includemismatch,
                verbose=options.verbose)

    # run the existing tool neuron-dump.py as well
    run_neuron_dump(outdir=options.outdir, verbose=options.verbose)

    package_tarball(outdir=options.outdir, allowmodel=options.allowmodel,
                    ccdir=options.ccdir, verbose=options.verbose)

    # change permissions for the directory and output
    os.system("chown -R {} {}".format(os.getlogin(), os.path.split(options.outdir)[0]))

    # write_miscinfo(outdir=options.outdir, data=l3)


if __name__ == "__main__":

    main()
