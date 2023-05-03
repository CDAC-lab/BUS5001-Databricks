# Databricks notebook source
# MAGIC %md
# MAGIC # New York Taxi Trips Dataset Analysis
# MAGIC The [NYC Taxi and Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/index.page) has publicly released a dataset of taxi trips from January 2009 to date. Trip data is published monthly (with two months delay). 
# MAGIC
# MAGIC The dataset constitutes a publicly available big data datasets, includes >100GB and more than 2 billion records. There are 4 different types of trip data available including yellow, green, for-hire and high volume, we will be focusing on the data published for the yellow taxis. 
# MAGIC
# MAGIC The data dictionary can be found [here](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
# MAGIC
# MAGIC Dataset Download Link: [link](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) 
# MAGIC
# MAGIC Getting started with Databricks notebook: [link](https://docs.microsoft.com/en-us/azure/databricks/notebooks/notebooks-use)
# MAGIC <br>
# MAGIC
# MAGIC We will perform following operations on the dataset.
# MAGIC 1. Extract - Load - Transform Process
# MAGIC 2. Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing libraries

# COMMAND ----------

import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType,DecimalType
from pyspark.sql.functions import pandas_udf, PandasUDFType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exracting / Ingesting Data
# MAGIC
# MAGIC The source files are stored in parquet format with the following naming conventions; https[:]//d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_*{YYYY}-{MM}*.parquet
# MAGIC
# MAGIC We will use a bash script to iterate through the necessary years and months to download the taxi datasets related to 2018 and 2019 from NYC TLC S3 bucket.

# COMMAND ----------

# %sh
# for year in {2018..2019}
# do
#   for month in {01..12}
#   do
#     wget -P /dbfs/tmp/taxi-csv/ "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$year-$month.parquet"
#   done
# done


# COMMAND ----------

# Azure storage access info
blob_account_name = "azureopendatastorage"
blob_container_name = "nyctlc"
blob_relative_path = "yellow"
blob_sas_token = "r"

# Allow SPARK to read from Blob remotely
# Uses string interpolation https://stackabuse.com/python-string-interpolation-with-the-percent-operator/ for string formatting
wasbs_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)
spark.conf.set(
  'fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name),
  blob_sas_token)
print('Remote blob path: ' + wasbs_path)

# COMMAND ----------

# SPARK read parquet, note that it won't load any data yet by now
df = spark.read.parquet(wasbs_path)

print('Register the DataFrame as a SQL temporary view: source')
df.createOrReplaceTempView('source')

# COMMAND ----------

# Display top 10 rows
print('Displaying top 10 rows: ')
display(spark.sql('SELECT * FROM source LIMIT 10'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform Data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In this section we will
# MAGIC - rename columns to a more consistant and meaningfull format
# MAGIC - transform datetime strings to unix timestamp type
# MAGIC - transform decimal values to Double type

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data dictionary
# MAGIC ```
# MAGIC vendor_name:string           --> vendor_id string
# MAGIC tpep_pickup_datetime:string  --> pickup_datetime datetime
# MAGIC tpep_dropoff_datetime:string --> dropoff_datetime datetime
# MAGIC passenger_Count:string       --> passenger_count integer
# MAGIC trip_distance:string         --> trip_distance double
# MAGIC pickup_location_id:string    --> pickup_location_id double
# MAGIC dropoff_location_id:string   --> dropoff_location_id double
# MAGIC RateCodeID:string            --> rate_code_id string
# MAGIC store_and_fwd_flag:string    --> store_and_forward string
# MAGIC payment_type:string          --> payment_type string
# MAGIC fare_amount:string           --> fare_amount double
# MAGIC surcharge:string             --> extra double
# MAGIC mta_tax:string               --> mta_tax double
# MAGIC tip_amount:string            --> tip_amount double
# MAGIC tolls_amount:string          --> tolls_amount double
# MAGIC total_amount:string          --> total_amount double
# MAGIC
# MAGIC ```

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

df_yellow_taxi_data = df.withColumnRenamed("vendorID","vendor_id")\
.withColumn("pickup_datetime",from_unixtime(unix_timestamp(col("tpepPickupDateTime"))))\
.withColumn("dropoff_datetime",from_unixtime(unix_timestamp(col("tpepDropoffDateTime"))))\
.withColumnRenamed("storeAndFwdFlag","store_and_fwd_flag")\
.withColumnRenamed("RatecodeID","rate_code_id")\
.withColumn("pickup_location_id",col("puLocationId").cast(DoubleType()))\
.withColumn("dropoff_location_id",col("doLocationId").cast(DoubleType()))\
.withColumn("passenger_count",col("passengerCount").cast(IntegerType()))\
.withColumn("trip_distance",col("tripDistance").cast(DoubleType()))\
.withColumn("fare_amount",col("fareAmount").cast(DoubleType()))\
.withColumn("extra",col("improvementSurcharge").cast(DoubleType()))\
.withColumn("mta_tax",col("mtaTax").cast(DoubleType()))\
.withColumn("tip_amount",col("tipAmount").cast(DoubleType()))\
.withColumn("tolls_amount",col("tollsAmount").cast(DoubleType()))\
.withColumn("total_amount", col("totalAmount").cast(DoubleType()))\
.withColumn("payment_type",col("paymentType"))\
.select(["vendor_id","pickup_datetime","dropoff_datetime","store_and_fwd_flag","rate_code_id","pickup_location_id","dropoff_location_id","passenger_count","trip_distance", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "total_amount", "payment_type"])


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Let's filter the dataset to only contain dates from 2018 and 2019

# COMMAND ----------

df_yellow_18_19 = df_yellow_taxi_data.filter((df_yellow_taxi_data.pickup_datetime >= '2018-01-01') & (df_yellow_taxi_data.pickup_datetime <= '2019-12-31'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create delta table

# COMMAND ----------

df_yellow_18_19.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/data/nyc-yellow/2018")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize delta tables

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC OPTIMIZE delta. `dbfs:/data/nyc-yellow/2018`
# MAGIC ZORDER BY (pickup_datetime)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL delta. `dbfs:/data/nyc-yellow/2018`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM delta. `dbfs:/data/nyc-yellow/2018`;
