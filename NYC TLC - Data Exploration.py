# Databricks notebook source
# MAGIC %md
# MAGIC # New York Taxi Trips Dataset Analysis
# MAGIC The NYC Taxi and Limousine Commission (TLC) has publicly released a dataset of taxi trips from January 2009 to date (June 22 as of September). Trip data is published monthly (with two months delay). 
# MAGIC
# MAGIC The dataset forms one of the few publicly available big data datasets, includes >100GB and more than 2 billion records. There are 4 different types of trip data available including yellow, green, for-hire and high volume, we will be focusing on the data published for the yellow taxis. 
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

import geopandas as gpd

# from branca.colormap import linear
from shapely.geometry import Point, Polygon, shape
from shapely import wkb, wkt
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType,DecimalType
from pyspark.sql.functions import pandas_udf, PandasUDFType
import shapely.speedups
shapely.speedups.enable() # this makes some spatial queries run faster

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Loading / Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Download and Load Taxi Zone related data

# COMMAND ----------

df_csv = pd.read_csv("https://bus5001.blob.core.windows.net/processed/taxi_zones.csv")
spark_df = spark.createDataFrame(df_csv).cache()
spark_df.createOrReplaceTempView('taxiGeom')


# COMMAND ----------

taxi_zones_df = pd.read_csv("https://bus5001.blob.core.windows.net/processed/taxi+_zone_lookup.csv")
taxiZonesDF=spark.createDataFrame(taxi_zones_df).cache()
taxiZonesDF.createOrReplaceTempView('taxiZones')

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM taxiGeom

# COMMAND ----------

display(taxiZonesDF)

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

df_yellow_18_19 = spark.read.load("/data/nyc-yellow/2018")

# COMMAND ----------

from pyspark.sql.functions import spark_partition_id, asc, desc
df_yellow_18_19\
    .withColumn("partitionId", spark_partition_id())\
    .groupBy("partitionId")\
    .count()\
    .orderBy(asc("count"))\
    .show()

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL delta. `dbfs:/data/nyc-yellow/2018`

# COMMAND ----------

df_yellow_18_19.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now you can run SQL queries on top of the temporary table and delta table you created. Also you can use the Spark API to query as well.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The first 10 rows using a SQL query

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total trip count

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM delta. `dbfs:/data/nyc-yellow/2018`;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip count by date using Spark API

# COMMAND ----------

df_yellow_18_19.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC    COUNT(pickup_datetime) trip_count
# MAGIC   ,to_date(pickup_datetime) date
# MAGIC FROM delta. `dbfs:/data/nyc-yellow/2018`
# MAGIC GROUP BY
# MAGIC    to_date(pickup_datetime)
# MAGIC ORDER BY
# MAGIC   to_date(date)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Payment type as a percentage over time 

# COMMAND ----------

display(spark.sql(
    '''
    SELECT
       COUNT(*) trips
      ,to_date(pickup_datetime) date
      ,CASE
        WHEN payment_type = 1 THEN 'Credit card'
        WHEN payment_type = 2 THEN 'Cash'
        WHEN payment_type = 3 THEN 'No charge'
        WHEN payment_type = 4 THEN 'Dispute'
        WHEN payment_type = 5 THEN 'Unknown'
        ELSE 'Voided trip'
      END AS Payment_type
    FROM delta. `dbfs:/data/nyc-yellow/2018`
    GROUP BY
       to_date(pickup_datetime)
      ,payment_type
    ORDER BY
      to_date(date)
    '''
),False)

# COMMAND ----------

# MAGIC %md
# MAGIC US Holidays in 2018 and 2019
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC New Year's Day	                     Mon, 1 Jan 2018
# MAGIC Martin Luther King Jr. Day	         Mon, 15 Jan 2018
# MAGIC Memorial Day	                     Mon, 28 May 2018
# MAGIC Independence Day	                 Wed, 4 July 2018
# MAGIC Labor Day	                         Mon, 3 Sept 2018
# MAGIC Veterans Day	                     Mon, 12 Nov 2018
# MAGIC Thanksgiving	                     Thu, 22 Nov 2018
# MAGIC George H. W. Bush Memorial Day	     Wed, 5 Dec 2018
# MAGIC Christmas Day	                     Tue, 25 Dec 2018
# MAGIC
# MAGIC
# MAGIC New Year's Day	                     Tue, 1 Jan 2019
# MAGIC Martin Luther King Jr. Day	         Mon, 21 Jan 2019
# MAGIC Memorial Day	                     Mon, 27 May 2019
# MAGIC Independence Day	                 Thu, 4 July 2019
# MAGIC Labor Day	                         Mon, 2 Sept 2019
# MAGIC Veterans Day	                     Mon, 11 Nov 2019
# MAGIC Thanksgiving	                     Thu, 28 Nov 2019
# MAGIC Christmas Day	                     Wed, 25 Dec 2019
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Taxi trip pickups by taxi zone 

# COMMAND ----------

pickup_by_zone = spark.sql(
'''
SELECT 
   tg.zone as Zone
  ,t.pickup_location_id as pickup_location_id
  ,tg.the_geom as geometry
  ,t.trip_count
FROM
taxiGeom tg INNER JOIN (
    SELECT
        pickup_location_id,
        COUNT(pickup_location_id) as trip_count
    FROM delta. `dbfs:/data/nyc-yellow/2018`
    GROUP BY pickup_location_id
) t 
ON t.pickup_location_id = tg.LocationID 
ORDER BY t.pickup_location_id
'''
).toPandas()

# COMMAND ----------

pickup_by_zone_gs = gpd.GeoSeries.from_wkt(pickup_by_zone['geometry'])
pickup_by_zone_gdf = gpd.GeoDataFrame(pickup_by_zone, geometry=pickup_by_zone_gs, crs="EPSG:4326")

m = pickup_by_zone_gdf.explore(
    column="trip_count",
    tooltip=["Zone","trip_count"],
    legend=True,
    legend_kwds=dict(colorbar=False),
    popup=True, # show all values in popup (on click)
    tiles="CartoDB positron", # use "CartoDB positron" tiles
    cmap='YlOrBr',
    scheme='quantiles',
)
m 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Taxi trip dropoffs by taxi zone 

# COMMAND ----------

dropoff_by_zone = spark.sql(
'''
SELECT 
 tg.zone as Zone
 ,t.dropoff_location_id as dropoff_location_id
,tg.the_geom as geometry
,t.trip_count
FROM
taxiGeom tg INNER JOIN (
    SELECT
        dropoff_location_id,
        COUNT(dropoff_location_id) as trip_count
    FROM delta. `dbfs:/data/nyc-yellow/2018`
    GROUP BY dropoff_location_id
) t 
ON t.dropoff_location_id = tg.LocationID 
ORDER BY t.dropoff_location_id
'''
).toPandas()

# COMMAND ----------

dropoff_by_zone_gs = gpd.GeoSeries.from_wkt(dropoff_by_zone['geometry'])
dropoff_by_zone_gdf  = gpd.GeoDataFrame(dropoff_by_zone, geometry=dropoff_by_zone_gs, crs="EPSG:4326")

m = dropoff_by_zone_gdf.explore(
    column="trip_count",
    tooltip=["Zone","trip_count"],
    legend=True,
    legend_kwds=dict(colorbar=False),
    popup=True, # show all values in popup (on click)
    tiles="CartoDB positron", # use "CartoDB positron" tiles
    cmap='YlOrBr',
    scheme='quantiles',
)
m

# COMMAND ----------

# MAGIC %md
# MAGIC ## Taxi trip pickups by taxi zone over time

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC    tg.zone as Zone
# MAGIC   ,t.pickup_location_id as pickup_location_id
# MAGIC   ,st_geomFromWKT(tg.the_geom) as geometry
# MAGIC   ,t.trip_count
# MAGIC   ,t.trip_date
# MAGIC FROM
# MAGIC taxiGeom tg INNER JOIN (
# MAGIC     SELECT
# MAGIC         pickup_location_id
# MAGIC         ,to_date(pickup_datetime) trip_date
# MAGIC         ,COUNT(pickup_location_id) as trip_count
# MAGIC     FROM delta. `dbfs:/data/nyc-yellow/2018` WHERE pickup_datetime BETWEEN "2018-01-01" AND "2018-02-01"
# MAGIC     GROUP BY 
# MAGIC         to_date(pickup_datetime)
# MAGIC         ,pickup_location_id
# MAGIC ) t 
# MAGIC ON t.pickup_location_id = tg.LocationID 
# MAGIC ORDER BY t.pickup_location_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip count by passenger count

# COMMAND ----------

display(
    df_yellow_18_19\
    .groupBy(col("passenger_count").alias("passenger_count"))\
    .agg(count("passenger_count").alias("trip count"))\
    .orderBy(col('passenger_count'))\
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Trip count by weekday

# COMMAND ----------

display(
    df_yellow_18_19\
    .groupBy(date_format(to_date("pickup_datetime"),"EEEE").alias("day"),dayofweek(to_date("pickup_datetime")).alias("day_number"))\
    .agg(count("pickup_datetime").alias("trip count"))\
    .orderBy(col("day_number"))\
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Taxi trip pickups by hour

# COMMAND ----------

display(
    df_yellow_18_19\
    .groupBy(hour("pickup_datetime").alias("hour"))\
    .agg(count("pickup_datetime").alias("pickups"))\
    .orderBy(col("hour"))\
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Taxi trip dropoffs by hour

# COMMAND ----------

display(
    df_yellow_18_19\
    .groupBy(hour("dropoff_datetime").alias("hour"))\
    .agg(count("dropoff_datetime").alias("dropoffs"))\
    .orderBy(col("hour"))\
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Taxi trip origin destination heatmap

# COMMAND ----------

pickup_dropoff_heatmap = spark.sql('''
SELECT
   pu.Zone AS pickup_location
  ,do.Zone AS dropoff_location
  ,t.trip_count
FROM
(
    (
    SELECT 
       pickup_location_id AS pickup_location
      ,dropoff_location_id AS dropoff_location
      ,count(pickup_location_id) AS trip_count 
    FROM delta. `dbfs:/data/nyc-yellow/2018` 
        WHERE 
        pickup_location_id < 264 
        AND 
        dropoff_location_id < 264
    GROUP BY 
        pickup_location_id
        ,dropoff_location_id
    ) t
    LEFT JOIN taxiZones pu ON t.pickup_location=pu.LocationID
    LEFT JOIN taxiZones do ON t.dropoff_location=do.LocationID
)
ORDER BY
    t.trip_count DESC
    ,pu.Borough DESC
    ,do.Borough DESC
LIMIT 100
    '''
).toPandas()

# COMMAND ----------

pickup_dropoff_heatmap_pivot = pd.pivot_table(pickup_dropoff_heatmap,columns='dropoff_location',index='pickup_location')
pickup_dropoff_heatmap_pivot[pickup_dropoff_heatmap_pivot<1000] = np.nan

# COMMAND ----------

sns.set(rc={'figure.figsize':(15,15)})
sns.heatmap(pickup_dropoff_heatmap_pivot,cmap='rocket_r')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip duration by pickup hour

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   hour(pickup_datetime) AS pickup_hour,
# MAGIC   AVG(
# MAGIC     (bigint(to_timestamp(dropoff_datetime))) - (bigint(to_timestamp(pickup_datetime)))
# MAGIC   ) AS trip_duration
# MAGIC FROM
# MAGIC   delta. `dbfs:/data/nyc-yellow/2018`
# MAGIC GROUP BY
# MAGIC   hour(pickup_datetime)
# MAGIC ORDER BY
# MAGIC   pickup_hour

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip duration by dropoff hour

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   hour(dropoff_datetime) AS dropoff_hour,
# MAGIC   AVG(
# MAGIC     (bigint(to_timestamp(dropoff_datetime))) - (bigint(to_timestamp(pickup_datetime)))
# MAGIC   ) AS trip_duration
# MAGIC FROM
# MAGIC   delta. `dbfs:/data/nyc-yellow/2018`
# MAGIC GROUP BY
# MAGIC   hour(dropoff_datetime)
# MAGIC ORDER BY
# MAGIC   dropoff_hour

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip duration by week day

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   date_format(pickup_datetime, "EEEE") AS weekday,
# MAGIC   dayofweek(pickup_datetime) as day_number,
# MAGIC   AVG(
# MAGIC     (bigint(to_timestamp(dropoff_datetime))) - (bigint(to_timestamp(pickup_datetime)))
# MAGIC   ) AS trip_duration
# MAGIC FROM
# MAGIC   delta. `dbfs:/data/nyc-yellow/2018`
# MAGIC GROUP BY
# MAGIC   date_format(pickup_datetime, "EEEE"),
# MAGIC   dayofweek(pickup_datetime)
# MAGIC ORDER BY
# MAGIC   day_number

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip duration by passenger count

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   passenger_count,
# MAGIC   AVG(
# MAGIC     (bigint(to_timestamp(dropoff_datetime))) - (bigint(to_timestamp(pickup_datetime)))
# MAGIC   ) AS trip_duration
# MAGIC FROM
# MAGIC   delta. `dbfs:/data/nyc-yellow/2018`
# MAGIC GROUP BY
# MAGIC   passenger_count
# MAGIC HAVING
# MAGIC   passenger_count < 15
# MAGIC ORDER BY
# MAGIC   passenger_count

# COMMAND ----------

# MAGIC %md
# MAGIC ## Average trip fare over time

# COMMAND ----------

display(spark.sql(
    '''
    SELECT
       AVG(total_amount) avg_total_amount
      ,to_date(pickup_datetime) date
    FROM delta. `dbfs:/data/nyc-yellow/2018` WHERE total_amount >= 0
    GROUP BY
       to_date(pickup_datetime)
    ORDER BY
      to_date(date)
    '''
),False)

# COMMAND ----------

# MAGIC %md
# MAGIC [Price Hike](https://www.ny1.com/nyc/all-boroughs/news/2019/02/03/congestion-pricing-surcharge-in-nyc-goes-into-effect)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Average tip amount over time

# COMMAND ----------

display(spark.sql(
    '''
    SELECT
       AVG(tip_amount) avg_tip_amount
      ,to_date(pickup_datetime) date
    FROM delta. `dbfs:/data/nyc-yellow/2018` WHERE tip_amount >= 0
    GROUP BY
       to_date(pickup_datetime)
    ORDER BY
      to_date(date)
    '''
),False)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Tip to Total Amount percentage 

# COMMAND ----------

display(spark.sql(
    '''
    SELECT
       (AVG(tip_amount) / AVG(total_amount-tip_amount))* 100 tip_percentage_of_total_amount
      ,to_date(pickup_datetime) date
    FROM delta. `dbfs:/data/nyc-yellow/2018` WHERE tip_amount >= 0 AND total_amount >= 0
    GROUP BY
       to_date(pickup_datetime)
    ORDER BY
      to_date(date)
    '''
),False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Average trip distance over time

# COMMAND ----------

display(spark.sql(
    '''
    SELECT
       AVG(trip_distance) avg_trip_distance
      ,to_date(pickup_datetime) date
    FROM delta. `dbfs:/data/nyc-yellow/2018`
    GROUP BY
       to_date(pickup_datetime)
    ORDER BY
      to_date(date)
    '''
),False)
