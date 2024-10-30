.. _inf2-performance:

Inf2 Inference Performance
==========================

.. contents:: Table of contents
   :local:
   :depth: 1

*Last update: September 16th, 2024*

.. _inf2_inference_perf:

Encoder Models
--------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_encoder.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')
            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (inference/second)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)
            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/second)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Sequence Length', 'Model Data Type','Compilation Autocast Data Type', 'OS Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
            df['Throughput (inference/second)'] = df['Throughput (inference/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_encoder.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')
            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (inference/second)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)
            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/second)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Sequence Length', 'Model Data Type','Compilation Autocast Data Type', 'OS Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
            df['Throughput (inference/second)'] = df['Throughput (inference/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


Encoder-Decoder Models
----------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_encoder_decoder.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (tokens/second)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (tokens/second)', 'Latency per Token P50 (ms)', 'Latency per Token P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'TP Degree',	'DP Degree', 'Batch Size', 'Sequence Length', 'Input Length', 'Output Length', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

            df['Throughput (tokens/second)'] = df['Throughput (tokens/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency per Token P50 (ms)', 'Latency per Token P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         Only for Encoder-Decoder

         **Throughput (tokens/second)** counts both input and output tokens

         **Latency per Token** counts both input and output tokens


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_encoder_decoder.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (tokens/second)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (tokens/second)', 'Latency per Token P50 (ms)', 'Latency per Token P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'TP Degree',	'DP Degree', 'Batch Size', 'Sequence Length', 'Input Length', 'Output Length', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

            df['Throughput (tokens/second)'] = df['Throughput (tokens/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency per Token P50 (ms)', 'Latency per Token P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Throughput (tokens/second)** counts both input and output tokens

         **Latency per Token** counts both input and output tokens
        

Vision Transformers Models
--------------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_vision_transformers.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M images', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_vision_transformers.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework','Inst. Type','Task', 'Throughput (inference/sec)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M images','Application Type','Neuron Version','Run Mode','Batch Size','Model Data Type', 'Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


Convolutional Neural Networks (CNN) Models
------------------------------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_vision_cnn.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M images', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_vision_cnn.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework','Inst. Type','Task', 'Throughput (inference/sec)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M images','Application Type','Neuron Version','Run Mode','Batch Size','Model Data Type', 'Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


Stable Diffusion Models
-----------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_vision_sd.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M images', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Cost per 1M images** is calculated using RI-Effective hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_vision_sd.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework','Inst. Type','Task', 'Throughput (inference/sec)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M images','Application Type','Neuron Version','Run Mode','Batch Size','Model Data Type', 'Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Cost per 1M images** is calculated using RI-Effective hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.

Diffusion Transformer Models
----------------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_vision_dit.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M images', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Cost per 1M images** is calculated using RI-Effective hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_vision_dit.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M images'] = ((1.0e6 / df['Throughput (inference/sec)']) * (df['RI-Effective hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework','Inst. Type','Task', 'Throughput (inference/sec)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M images','Application Type','Neuron Version','Run Mode','Batch Size','Model Data Type', 'Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Cost per 1M images'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Cost per 1M images** is calculated using RI-Effective hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.




.. note::

      See :ref:`neuron_hw_glossary` for abbreviations and terms
