
import setuptools


setuptools.setup(
    name='bert-demo',
    version='2019.12.13',
    description='BERT Client-Server Demo',
    author='Amazon AWS',
    author_email='aws-neuron-support@amazon.com',
    license='BSD',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='bert',
    include_package_data=True,
    packages=setuptools.PEP420PackageFinder.find(),
    package_data={'': [
        '*',
    ]},
    entry_points={
        'console_scripts': [
            'neuron_bert_model=bert_demo.bert_model:main',
            'bert_server=bert_demo.bert_server:serve',
            'bert_client=bert_demo.bert_client:client',
        ],
    },
    install_requires=[
    ],
)
