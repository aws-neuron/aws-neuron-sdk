.. _trn1-inference-performance:

Trn1/Trn1n Inference Performance
================================

.. contents:: Table of contents
   :local:


*Last update:  May 20th, 2025*


.. _NLP:

Encoder Models
--------------

.. tab-set::

   .. tab-item:: Throughput optimized

      .. df-table::
         :header-rows: 1

         df = pd.read_csv('throughput_data_encoder.csv')
         df_prices = pd.read_csv('trn1_instance_prices.csv')
         df = pd.merge(df,df_prices,on='Inst. Type')

         df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

         cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size','Sequence Length', 'Model Data Type','Compilation Autocast Data Type','OS Type']
         df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

         df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
         int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
         df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


   .. tab-item:: Latency optimized

      .. df-table::
         :header-rows: 1

         df = pd.read_csv('latency_data_encoder.csv')
         df_prices = pd.read_csv('trn1_instance_prices.csv')
         df = pd.merge(df,df_prices,on='Inst. Type')

         df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

         cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size','Sequence Length', 'Model Data Type','Compilation Autocast Data Type','OS Type']
         df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

         df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
         int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
         df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

Encoder-Decoder Models
----------------------

.. tab-set::

   .. tab-item:: Throughput optimized

      .. df-table::
         :header-rows: 1

         df = pd.read_csv('throughput_data_encoder_decoder.csv')
         df_prices = pd.read_csv('trn1_instance_prices.csv')
         df = pd.merge(df,df_prices,on='Inst. Type')
         df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (tokens/second)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)
         cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (tokens/second)', 'Latency per Token P50 (ms)', 'Latency per Token P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'TP Degree',        'DP Degree', 'Batch Size', 'Sequence Length', 'Input Length', 'Output Length', 'Model Data Type','Compilation Autocast Data Type']
         df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
         df['Throughput (tokens/second)'] = df['Throughput (tokens/second)'].round(2).astype('float',copy=True)
         int_cols = ['Latency per Token P50 (ms)', 'Latency per Token P99 (ms)']
         df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

      .. note::
         Only for Encoder-Decoder

         **Throughput (tokens/second)** counts both input and output tokens

         **Latency per Token** counts both input and output tokens

         Applicable to all models

         **Cost per 1M inferences** is calculated using RI-Effective hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.


   .. tab-item:: Latency optimized

      .. df-table::
         :header-rows: 1

         df = pd.read_csv('latency_data_encoder_decoder.csv')
         df_prices = pd.read_csv('trn1_instance_prices.csv')
         df = pd.merge(df,df_prices,on='Inst. Type')
         df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (tokens/second)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)
         cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (tokens/second)', 'Latency per Token P50 (ms)', 'Latency per Token P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'TP Degree',        'DP Degree', 'Batch Size', 'Sequence Length', 'Input Length', 'Output Length', 'Model Data Type','Compilation Autocast Data Type']
         df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
         df['Throughput (tokens/second)'] = df['Throughput (tokens/second)'].round(2).astype('float',copy=True)
         int_cols = ['Latency per Token P50 (ms)', 'Latency per Token P99 (ms)']
         df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

      .. note::

         Only for Encoder-Decoder

         **Throughput (tokens/second)** counts both input and output tokens

         **Latency per Token** counts both input and output tokens


      .. note::

         **Cost per 1M inferences** is calculated using RI-Effective hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.
