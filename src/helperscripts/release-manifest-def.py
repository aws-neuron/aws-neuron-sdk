
neuron_releases={
    "repos":{
        "whl"="_url",           # url of the wheel repo
        "rpm"="_url",           # url of the rpm repo (yum)
        "deb"="_url",           # url of the debian repo (apt)
    }
    "manifest_date": "_date",
    "manifest_version":"_ver"   # Will increment when format change
    "latest_release":{
        "_instance":{           # can be "inf1", "trn1", etc.. 
            "version":"_ver"    # latest neuron release that support the _instance 
        }
    }
    "neuron_versions"={         # all neuron release versions supported by this manifest
        "_neuron_version":{     # Neuron release version entry e.g. "1.14.0"
            "python_ver": ["_ver"]              # list of python versions supported by this neuron release, e.g. "3.6"
            "instance_support": ["_instance"]   # list of instances supported by this neuron release
            "arch":["_arch"]                    # list of architectures supported by this neuron release (e.g. x86)
            "components":{                      # all components included in this neuron release 
                                                # (e.g. compiler, driver , pytorch ...)
                "_component_name":{             # component entry (e.g. driver, compiler)
                    "framework":_boolean        # is this component a framework ? 
                                                # needed since there is a differces in versioning and content etc .. 
                    "packages":{                # all packages of this component that included in this release 
                                                # e.g. mxnet support mx_neuron and mxnet-neuron
                        "_package_name":{       # package entry (e.g. mx_neuron)
                            "install_on_compute_instance":_booolean     # can this package installed on compute instance?
                            "versions":{                            # all versions of the specific package
                                                                    # e.g. torch-neuron may include multiple versions
                                "_ver":{                            # package version entry (e.g. 1.4.1.0)
                                    "pre_install_cmds":["_cmd"]     # a list of commands to call before installing
                                                                    # the package, e.g. when a plugin need to install the
                                                                    # framework first , as in mx_neuron
                                    "post_install_cmds":["_cmd"]    # a list of commands to call after installaing the package
                                    "format":["_format"]            # package format (e.g. bin or src)
                                    "content":["_content"]          # package content 
                                                                    # (e.g. tools include neuron-top, neuron monitor etc .. )
                                    "package_type":["_type"]        # list of package type supported ( e.g. whl, rpm, deb)
                                }
                            }                     
                        }
                    }
                }
            }
        }
    },
    "softwarelifecycle":{           # Status of neuron software releases (supported, maintained, deprecated)
                                    # Releases that are not under "supported" or "maintained" should be "supported"
        "maintained":{              # Releases that are being maintained, no active development, bug fixes can be provided
                                    # releases can be Neuron release, component (e.g. runtime), or a framework (e.g. pytorch-1.5.x)
            "neuron_versions":{     # Neuron versions that are under maintanance status
                "from":"_ver"       # from neuron release version
                "to":"_ver"         # to neuron release version
            },
            "components":{              # Components that are under maintanance status
                "_component_name":{     # packages in that component
                    "_package_name":{   # package entry
                        "from":"_ver"   # from version
                        "to":"_ver"     # to version
                    }
                }

            },
            "frameworks":{              # Frameworks that are under maintanance status
                "pytorch":{             # Pytorch versions that are under maintanance status
                    "from":"_ver"       # from version
                    "to":"_ver"         # to version
                },
                "tensorflow":{          # Pytorch versions that are under maintanance status
                    "from":"_ver"       # from version
                    "to":"_ver"         # to version
                },
                "mxnewt":{              # MXNet versions that are under maintanance status
                    "from":"_ver"       # from version
                    "to":"_ver"         # to version
                }
            }
        },
        "deprecated":{                  # Releases that are deprecated, no bug fixes
                                        # format similar to "maintained" section
        },
    },
    "compatability": {                  # compatability section
        "_component_name": {            # component entry
            "_package_name": {          # package entry
                "_ver_to__ver": {       # compatability entry
                    "from": "_ver",     # from version
                    "to": "_ver",       # to version
                    "instance_support": [   # instance compatability
                        "_instance"
                    ],
                    "arch": [               # arch compatability
                        "_arch"
                    ],
                    "components": {                 # components compatability section 
                        "_component_name": {        # component entry
                            "_package_name": {      # package entry
                                "from": "_ver",     # from version
                                "to": "_ver"        # to version
                            }
                        }
                    }
                }
            }
        }
    }
}
