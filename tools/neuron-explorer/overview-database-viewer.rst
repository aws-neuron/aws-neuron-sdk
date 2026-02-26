.. meta::
    :description: Learn about the Database Viewer tool in Neuron Explorer for querying and exploring profiling data using SQL or natural language queries.
    :date-modified: 01/27/2026

.. _database-viewer-overview:

Database Viewer
=====================

The Database Viewer offers an interactive interface providing visibility to all underlying data that the Neuron Explorer
processes from a :doc:`NEFF </neuron-runtime/explore/work-with-neff-files>` and NTFF. 
Use this tool to develop your own analyses, examine profiling data stored in database tables, or run ad-hoc queries during performance analysis. 
You can access this data through natural language queries or raw SQL.


.. image:: /tools/profiler/images/database-viewer.png

Table Selection and Schema Inspection
-------------------------------------

When the tool loads, it fetches the list of available database tables. Select a table from the dropdown to view its schema.

The schema table displays:

* **Field Name** - Column name (hover for description tooltip).
* **Data Type** - The data type of the field.
* **Required** - Whether the field is required.
* **Unit** - Measurement unit (if applicable).
* **Example** - Example value for the field.

Querying Data
-------------

The query input supports two modes:

1. **SQL queries** - Write standard SQL starting with ``SELECT``.
2. **Natural language queries** - Describe what you want in plain English.

Examples:

Natural language query to get the first 5 rows::

    Get the first 5 rows

SQL query to filter with conditions::

    SELECT field_name FROM table_name WHERE condition

Press **Enter** or click **Execute Query** to run. Use **Shift+Enter** for multi-line input.

Query Results
-------------

Results appear below the query input in reverse chronological order (newest first). Each result shows:

* The original query text.
* The generated SQL (for natural language queries).
* A scrollable results table.

Click **Export CSV** to download any result set as a CSV file.

.. image:: /tools/profiler/images/database-viewer-query-result.png