.. _inf2-performance:

Inf2 Performance
================

.. contents:: Table of contents
   :local:
   :depth: 1

*Last update: Apr 12th, 2023*

.. _inf2_inference_perf:

Language Models Inference Performance
---------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_language.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (inference/second)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/second)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Sequence Length', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

            df['Throughput (inference/second)'] = df['Throughput (inference/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_language.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (inference/second)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework','Inst. Type', 'Task', 'Throughput (inference/second)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M inferences','Application Type','Neuron Version','Run Mode','Batch Size','Sequence Length','Model Data Type', 'Compilation Autocast Data Type']

            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
            df['Throughput (inference/second)'] = df['Throughput (inference/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


Large Language Models Inference Performance
-------------------------------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_LLM.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (tokens/second)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (tokens/second)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'TP Degree',	'DP Degree', 'Batch Size', 'Sequence Length', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

            df['Throughput (tokens/second)'] = df['Throughput (tokens/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_LLM.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (tokens/second)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework','Inst. Type', 'Task', 'Throughput (tokens/second)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M inferences','Application Type','Neuron Version','Run Mode','TP Degree',	'DP Degree', 'Batch Size','Sequence Length','Model Data Type', 'Compilation Autocast Data Type']

            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

            df['Throughput (tokens/second)'] = df['Throughput (tokens/second)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


Vision Models Inference Performance
---------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data_vision.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Inferences per dollar'] = ((df['Throughput (inference/sec)'] * 3.6e3) / df['On-Demand hourly rate']).map('{:,.0f}/$'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework', 'Inst. Type', 'Task', 'Throughput (inference/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Inferences per dollar', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Inferences per dollar'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Cost per 1M inferences** is calculated using On-Demand hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.



    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data_vision.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Inferences per dollar'] = ((df['Throughput (inference/sec)'] * 3.6e3) / df['On-Demand hourly rate']).map('{:,.0f}/$'.format)

            cols_to_show = ['Model','Image Size','Scripts','Framework','Inst. Type','Task', 'Throughput (inference/sec)','Latency P50 (ms)','Latency P99 (ms)','Inferences per dollar','Application Type','Neuron Version','Run Mode','Batch Size','Model Data Type', 'Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Image Size', 'Inferences per dollar'])

            df['Throughput (inference/sec)'] = df['Throughput (inference/sec)'].round(2).astype('float',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)

        .. note::
         **Cost per 1M inferences** is calculated using On-Demand hourly rate.

         **Real Time** application refers to batch size 1 inference for minimal latency. **Batch** application refers to maximum throughput with minimum cost-per-inference.


.. note::

      See :ref:`neuron_hw_glossary` for abbreviations and terms
