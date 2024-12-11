from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, when, dayofmonth
from pyspark.ml.feature import StringIndexer, OneHotEncoder

def replace_date_with_seasons(df):

    #create month column
    df = df.withColumn("month", month(col("Date")))
    df = df.withColumn("day", dayofmonth(col("Date")))
    
    #define seasons fields
    df = df.withColumn("Spring", when(
        ((col("month") == 3) & (col("day") >= 21)) | 
        (col("month").isin(4, 5)) | 
        ((col("month") == 6) & (col("day") <= 20)), 1).otherwise(0))

    df = df.withColumn("Summer", when(
        ((col("month") == 6) & (col("day") >= 21)) | 
        (col("month").isin(7, 8)) | 
        ((col("month") == 9) & (col("day") <= 22)), 1).otherwise(0))

    df = df.withColumn("Autumn", when(
        ((col("month") == 9) & (col("day") >= 23)) | 
        (col("month").isin(10, 11)) | 
        ((col("month") == 12) & (col("day") <= 20)), 1).otherwise(0))

    df = df.withColumn("Winter", when(
        ((col("month") == 12) & (col("day") >= 21)) | 
        (col("month").isin(1, 2)) | 
        ((col("month") == 3) & (col("day") <= 19)), 1).otherwise(0))
    
    #remove Date, month and day columns
    
    df_seasons = df.drop("month", "day", "Date")

    return df_seasons


def encode_wit_one_hot(df, col_name):
    
    #add index column
    index_col = col_name + "Index"

    indexer = StringIndexer(inputCol=col_name, outputCol = index_col)
    df_indexed = indexer.fit(df).transform(df)
    
    #apply one hot encoding
    vector_col = col_name + "Vector"

    encoder = OneHotEncoder(inputCol=index_col, outputCol=vector_col)
    df_encoded = encoder.fit(df_indexed).transform(df_indexed)

    return df_encoded


def remove_outliers_with_IQR(df, features):

    boundaries = {}

    for feature in features:

        quantiles = df.approxQuantile(feature, [0.25, 0.75], 0.01)
        Q1 = quantiles[0]
        Q3 = quantiles[1]

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q1 + 1.5 * IQR

        boundaries[feature] = (lower_limit, upper_limit)

    #create filter
    filter_cond = " AND ".join( [ f'(`{feature}` >= {boundaries[feature][0]} AND `{feature}` <= {boundaries[feature][1]})' for feature in features ] )
    
    #filter dataset from outliers
    df_filtered = df.filter(filter_cond)

    return df_filtered

    


if __name__ == '__main__' :

    #LOADING SPARK SESSION AND DATA

    spark = SparkSession.builder.appName("avocado_price_prediction").getOrCreate()
    df = spark.read.csv("Augmented_avocado.csv", sep = ",", header=True, inferSchema = True)

    #PREPROCESSING
    df = df.drop("Unnamed: 0")

    df = replace_date_with_seasons(df)
    df = encode_wit_one_hot(df,"type")
    df = encode_wit_one_hot(df, "region")

    df = df.drop("type", "region")

    #zastanwić się czy rok też wyrzucić

    df = df.drop("4046", "4225", "4770", "Small Bags", "Large Bags", "XLarge Bags")

    features = ["Total Volume", "Total Bags"]
    df = remove_outliers_with_IQR(df, features)


    #TRAIN TEST SPLIT

    X = df.drop("AveragePrice")
    y = df.select("AveragePrice")

    spark.sparkContext.stop()

    #df.show(10)

