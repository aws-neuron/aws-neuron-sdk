.. _appnote-performance-benchmark:

Neuron Inference Performance
===================================

.. contents::
   :local:

The following tables contain the reference inference performance for models in the :ref:`neuron-tutorials`. Follow the links on each row to replicate similar results in your own environment. Refer to :ref:`ec2-then-ec2-setenv` documentation to create a new environment based on the latest Neuron release.

*Last update: March 25th, 2022*


.. _NLP:

Natural Language Processing
---------------------------

.. df-table::
   :header-rows: 1

   df = pd.read_csv('neuronperf_nlp.csv')
   df_prices = pd.read_csv('instance_prices.csv')
   df = pd.merge(df,df_prices,on='Inst. Type')

   df['Cost per 1M inferences'] = ((1.0e6 / df['Avg Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

   cols_to_show = ['Model', 'Scripts', 'Framework', 'Inst. Type', 'Avg Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model details' ]
   df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

   int_cols = ['Avg Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)']
   df[int_cols] = df[int_cols].round(0).astype('int',copy=True)


\*\ *Throughput and latency numbers in this table were computed using* NeuronPerf_\ *. To reproduce these results, install NeuronPerf and run the provided scripts.*

.. _NeuronPerf: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuronperf/index.html

.. df-table::
   :header-rows: 1

   df = pd.read_csv('data.csv')
   df_prices = pd.read_csv('instance_prices.csv')
   df = pd.merge(df,df_prices,on='Inst. Type').query('`Application`=="NLP"')

   df['Cost per 1M inferences'] = ((1.0e6 / df['Avg Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

   cols_to_show = ['Model', 'Tutorial', 'Framework', 'Inst. Type', 'Avg Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model details' ]
   df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

   int_cols = ['Avg Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)']
   df[int_cols] = df[int_cols].round(0).astype('int',copy=True)

\*\ *Throughput and latency numbers in this table were generated using Neuron Tutorials.*

Computer Vision
---------------

.. df-table::
   :header-rows: 1

   df = pd.read_csv('data.csv')
   df_prices = pd.read_csv('instance_prices.csv')
   df = pd.merge(df,df_prices,on='Inst. Type').query('`Application`=="CV"')

   df['Cost per 1M inferences'] = ((1.0e6 / df['Avg Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

   cols_to_show = ['Model', 'Tutorial', 'Framework', 'Inst. Type', 'Avg Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model details' ]
   df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences']).groupby('Model').head(2)

   int_cols = ['Avg Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)']
   df[int_cols] = df[int_cols].round(0).astype('int',copy=True)

\*\ *Throughput and latency numbers in this table were generated using Neuron Tutorials.*

.. note::
   **Cost per 1M inferences** is calculated using US East (N. Virginia) On-Demand hourly rate.

   **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.
