.. _trn1-performance:

Trn1 Performance
===========================

.. contents:: Table of contents
   :local:


*Last update:  February 8th, 2023*


.. _NLP:

Training Performance
--------------------

.. csv-table::
  :file: nlp_data.csv
  :header-rows: 1

Inference Performance
---------------------

.. tab-set::

   .. tab-item:: Throughput optimized
   
      .. df-table::
         :header-rows: 1

         df = pd.read_csv('throughput_data.csv')
         df_prices = pd.read_csv('trn1_instance_prices.csv')
         df = pd.merge(df,df_prices,on='Inst. Type')

         df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

         cols_to_show = ['Model', 'Framework', 'Inst. Type', 'Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model details' ]
         df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
         
         df['Throughput (/sec)'] = df['Throughput (/sec)'].round(0).astype('int',copy=True)
         int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
         df[int_cols] = df[int_cols].round(2).astype('float',copy=True)   
   
      .. note::
         **Throughput optimization was performed by selecting a batch size which maximized the metric. All compiler flags, data types, and parameters are identical between model configurations**
         **Cost per 1M inferences** is calculated using On-Demand hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.
   
   .. tab-item:: Latency optimized

      .. df-table::
         :header-rows: 1

         df = pd.read_csv('latency_data.csv')
         df_prices = pd.read_csv('trn1_instance_prices.csv')
         df = pd.merge(df,df_prices,on='Inst. Type')

         df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

         cols_to_show = ['Model', 'Framework', 'Inst. Type', 'Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model details' ]
         df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

         df['Throughput (/sec)'] = df['Throughput (/sec)'].round(0).astype('int',copy=True)
         int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
         df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

      .. note::
         **Latency optimization was performed by selecting a batch size which maximized the metric. All compiler flags, data types, and parameters are identical between model configurations**
         **Cost per 1M inferences** is calculated using On-Demand hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.


