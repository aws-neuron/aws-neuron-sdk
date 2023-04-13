.. _inf2-performance:

Inf2 Performance
================

.. contents:: Table of contents
   :local:
   :depth: 1

*Last update: Apr 12th, 2023*

.. _inf2_inference_perf:

Inference Performance
---------------------

.. tab-set::

    .. tab-item:: Throughput optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('throughput_data.csv')
            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework', 'Inst. Type', 'Throughput (/sec)', 'Latency P50 (ms)', 'Latency P99 (ms)', 'Cost per 1M inferences', 'Application Type', 'Neuron Version', 'Run Mode', 'Batch Size', 'Model Data Type','Compilation Autocast Data Type']
            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])

            df['Throughput (/sec)'] = df['Throughput (/sec)'].round(0).astype('int',copy=True)
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


    .. tab-item:: Latency optimized

        .. df-table::
            :header-rows: 1

            df = pd.read_csv('latency_data.csv')

            df_prices = pd.read_csv('inf2_instance_prices.csv')
            df = pd.merge(df,df_prices,on='Inst. Type')

            df['Cost per 1M inferences'] = ((1.0e6 / df['Throughput (/sec)']) * (df['On-Demand hourly rate'] / 3.6e3 )).map('${:,.3f}'.format)

            cols_to_show = ['Model','Scripts','Framework','Inst. Type','Throughput (/sec)','Latency P50 (ms)','Latency P99 (ms)','Cost per 1M inferences','Application Type','Neuron Version','Run Mode','Batch Size','Model Data Type', 'Compilation Autocast Data Type']

            df = df[cols_to_show].sort_values(['Model', 'Cost per 1M inferences'])
            int_cols = ['Latency P50 (ms)', 'Latency P99 (ms)']
            df[int_cols] = df[int_cols].round(2).astype('float',copy=True)


.. note::

      See :ref:`neuron_hw_glossary` for abbreviations and terms
