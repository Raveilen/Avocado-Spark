from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, when, dayofmonth
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

'''
TO DO:
-- try regression and xgboost
- Add Cross-validation
- Improve Random Forest Regresion Results
- Prepare Model For XGBoost
- Compare results between models (choose better)
- Function for sampling data from dataset
'''

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

    # df = df.withColumn("Winter", when(
    #     ((col("month") == 12) & (col("day") >= 21)) | 
    #     (col("month").isin(1, 2)) | 
    #     ((col("month") == 3) & (col("day") <= 19)), 1).otherwise(0))
    
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
        upper_limit = Q3 + 1.5 * IQR

        boundaries[feature] = (lower_limit, upper_limit)

    #create filter
    filter_cond = " AND ".join( [ f'(`{feature}` >= {boundaries[feature][0]} AND `{feature}` <= {boundaries[feature][1]})' for feature in features ] )
    
    #filter dataset from outliers
    df_filtered = df.filter(filter_cond)

    return df_filtered

def random_forest_regression_test(df_train, df_eval):

    rf = RandomForestRegressor(featuresCol='features', labelCol='AveragePrice', numTrees=100, maxDepth=5, maxBins=70)
    rf_model = rf.fit(df_train)

    rf_pred = rf_model.transform(df_eval)

    rf_pred.show(10)

    evaluator_rmse = RegressionEvaluator(labelCol="AveragePrice", predictionCol="prediction", metricName="rmse")
    rf_rmse = evaluator_rmse.evaluate(rf_pred)

    evaluator_r2 = RegressionEvaluator(labelCol="AveragePrice")
    rf_r2 = evaluator_r2.evaluate(rf_pred, {evaluator_r2.metricName: "r2"})

    evaluator_mae = RegressionEvaluator(labelCol="AveragePrice", predictionCol="prediction", metricName="mae")
    rf_mae = evaluator_mae.evaluate(rf_pred)

    print(f"RandomForest RMSE: {rf_rmse}")
    print(f"RandomForest R2: {rf_r2}")
    print(f"RandomForest MAE: {rf_mae}")

    return (rf_rmse, rf_r2, rf_mae)
    


if __name__ == '__main__' :

    #LOADING SPARK SESSION AND DATA

    spark = SparkSession.builder.appName("avocado_price_prediction").getOrCreate()
    df = spark.read.csv("Augmented_avocado.csv", sep = ",", header=True, inferSchema = True)

    #PREPROCESSING
    df = df.drop("Unnamed: 0")

    df = replace_date_with_seasons(df)
    #df = encode_wit_one_hot(df,"type")
    #df = encode_wit_one_hot(df, "region")

    region_idx = StringIndexer(inputCol='region', outputCol='regionLabel')
    region_idx_model = region_idx.fit(df)
    df = region_idx_model.transform(df)

    type_idx = StringIndexer(inputCol='type', outputCol='typeLabel')
    type_idx_model = type_idx.fit(df)
    df = type_idx_model.transform(df)

    df = df.drop("type", "region")

    df = df.drop("4046", "4225", "4770", "Small Bags", "Large Bags", "XLarge Bags", "year")

    #df = df.select("AveragePrice", "Total Volume", "Total Bags")

    df.show(10)

    features = ["Total Volume", "Total Bags"]

    #df = remove_outliers_with_IQR(df, features)

    #TRAIN TEST SPLIT

    df_train, df_eval = df.randomSplit([0.8, 0.2], 42)

    #COMBINING FEATURES

    vect = VectorAssembler(inputCols=df.columns[1:], outputCol="features")
    df_train = vect.transform(df_train)
    df_eval = vect.transform(df_eval)

    df_train = df_train.select("AveragePrice", "features")
    df_train.show(10)

    #FEATURE SCALING

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaler_model = scaler.fit(df_train)
    df_train = scaler_model.transform(df_train)
    df_eval = scaler_model.transform(df_eval)

    df_train = df_train.select("AveragePrice", "features", "scaledFeatures")

    df_train.show(10)

    #RANDOM FOREST REGRESSOR MODEL

    results = random_forest_regression_test(df_train, df_eval)

    #XGBoost
    ...

    spark.sparkContext.stop()

    

