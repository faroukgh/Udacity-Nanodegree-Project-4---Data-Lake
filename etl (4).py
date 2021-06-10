import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['IAM']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['IAM']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.3") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # Get filepath to song data file from JSON file
    song_data = "{}*/*/*.json".format(input_data)
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(col("song_id"), col("title"), col("artist_id"), col("year"), col("duration")).distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet("{}songs/songs_tbale.parquet".format(output_data))

    # extract columns to create artists table
    artists_table = df.select(col("artist_id"), col("name"), col("location"), col("latitude"), col("longitude")).distinct() 
    
    # write artists table to parquet files
    artists_table.write.parquet("{}artist/artists_table.parquet".format(output_data))
    df.createOrReplaceTempView("song_df_table")


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = "{}*/*/*events.json".format(input_data)

    # read log data file
    df = spark.read.json(log_data).dropDuplicates()
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong").cache()

    # extract columns for users table    
    artists_table = df.select(col("artist_id"), col("name"), col("location"), col("latitued"), col("longitude")).distinct()
    
    # write users table to parquet files
    artists_table.write.parquet("{}artist/artists_table.parquet".format(output_data))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000), TimestampType())
    df = df.withColumn("start_time", get_timestamp(col("ts")))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: to_date(x), TimestampType())
    df = df.withColumn("start_time", get_timestamp(col("ts")))
    
    # extract columns to create time table
    df = df.withColumn("hour", hour("timestamp"))
    df = df.withColumn("day", dayofmonth("timestamp"))
    df = df.withColumn("month", month("timestamp"))
    df = df.withColumn("week", weekofyear("timestamp"))
    df = df.withColumn("weekday", dayofweek("timestamp"))
    
    time_table = df.select(col("start_time"), col("hour"), col("day"), col("week"), col("month"), col("year"), col("weekday")).distinct()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet("{}time/time_table.parquet".format(output_data))

    # read in song data to use for songplays table
    song_df = spark.sql("SELECT DISTINCT song_id, artist_id, artist_name FROM song_df_table") 

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, song_df.name == df.artist, "inner").distinct().select(col("start_time"), col("user_id"), col("level"), col("session_id"), col("location"), col("user_agent"), col("song_id"), col("artist_id")).withColumn("songplay_id", monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet("{}songplays_table.parquet".format(output_data))


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "https://aws-emr-resources-320788447950-us-east-1.s3.amazonaws.com/notebooks/e-6V8QN8BC5T6XF2DS52KSJF63I/testnotebook.ipynb"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
