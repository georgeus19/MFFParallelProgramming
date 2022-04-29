
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import sum

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DuplicitCitizens")\
        .getOrCreate()

    input_data_path = "/mnt/1/seznam.csv"
    output_path = "/mnt/1/output.csv"


    schema = StructType([
        StructField("givenName", StringType(), True),
        StructField("familyName", StringType(), True),
        StructField("id", StringType(), True),
        StructField("postalCode", StringType(), True),
    ])

    df = spark.read.csv(path=input_data_path, schema=schema, sep=",", header=False)

    df = df.select(df.givenName, df.familyName, df.postalCode.substr(0, 1).alias("region"))
    
    df = df.groupBy(["region", "givenName", "familyName"]).count()
    
    df = df.filter(df["count"] > 1)

    df = df.groupBy("region").agg(sum("count").alias("duplicitCitizens"))
    df = df.orderBy("region", ascending=True)

    output = ""
    for row in df.collect():
        output += str(row["region"])
        output += ","
        output += str(row["duplicitCitizens"])
        output += "\n"

    spark.stop()

    with open(output_path, "w") as output_file:
        output_file.write(output)