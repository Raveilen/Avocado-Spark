#imports
from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_fun;
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from enum import Enum

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class RegressionModels(Enum):
    LINEAR_REGRESSION = 1
    RANDOM_FOREST = 2
    XGBOOST = 3


#core variables
spark = SparkSession.builder.appName("avocado_price_prediction").getOrCreate()
df = spark.read.csv("Augmented_avocado.csv", sep = ",", header=True, inferSchema = True).drop("Unnamed: 0")
numerical_features = ["Total Volume", "4046", "4225", "4770", "Total Bags", "year", "month"]
features = numerical_features.copy().append("type_encoded")
model = RegressionModels.LINEAR_REGRESSION
batch_size = 12 #data portion which is given to Model to conclude results


'''
TO DO:
-- try regression and xgboost
- Add Cross-validation
- Improve Random Forest Regresion Results
- Prepare Model For XGBoost
- Compare results between models (choose better)
- Function for sampling data from dataset
'''

def linear_regression_data_preprocessing():

    #scale numerical features
    numerical_assembler = VectorAssembler(inputCols=numerical_features, outputCol="numerical_features")
    scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features", withMean=True, withStd=True)

    #encode categorical features
    indexer = StringIndexer(inputCol="type", outputCol="type_index")
    encoder = OneHotEncoder(inputCol="type_index", outputCol="type_encoded")

    #create features vector 
    final_assembler = VectorAssembler(inputCols=["scaled_numerical_features", "type_encoded"], outputCol="features")

    #create processing pipeline
    pipeline = Pipeline(stages=[numerical_assembler, scaler, indexer, encoder, final_assembler])

    preprocessed_df = pipeline.fit(df).transform(df)
    
    return preprocessed_df

    

def replace_date_with_seasons(df):

    #create month column
    df = df.withColumn("month", sql_fun.month(sql_fun.col("Date")))
    df = df.withColumn("day", sql_fun.dayofmonth(sql_fun.col("Date")))
    
    #define seasons fields
    df = df.withColumn("Spring", sql_fun.when(
        ((sql_fun.col("month") == 3) & (sql_fun.col("day") >= 21)) | 
        (sql_fun.col("month").isin(4, 5)) | 
        ((sql_fun.col("month") == 6) & (sql_fun.col("day") <= 20)), 1).otherwise(0))

    df = df.withColumn("Summer", sql_fun.when(
        ((sql_fun.col("month") == 6) & (sql_fun.col("day") >= 21)) | 
        (sql_fun.col("month").isin(7, 8)) | 
        ((sql_fun.col("month") == 9) & (sql_fun.col("day") <= 22)), 1).otherwise(0))

    df = df.withColumn("Autumn", sql_fun.when(
        ((sql_fun.col("month") == 9) & (sql_fun.col("day") >= 23)) | 
        (sql_fun.col("month").isin(10, 11)) | 
        ((sql_fun.col("month") == 12) & (sql_fun.col("day") <= 20)), 1).otherwise(0))

    # df = df.withColumn("Winter", sql_fun.when(
    #     ((sql_fun.col("month") == 12) & (sql_fun.col("day") >= 21)) | 
    #     (sql_fun.col("month").isin(1, 2)) | 
    #     ((sql_fun.col("month") == 3) & (sql_fun.col("day") <= 19)), 1).otherwise(0))
    
    #remove Date, month and day columns
    
    df_seasons = df.drop("month", "day", "Date")

    return df_seasons


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

    if(model == RegressionModels.LINEAR_REGRESSION):
    #Preprocessing
        df = df.withColumn("year", sql_fun.year("Date")).withColumn("month", sql_fun.month("Date"))
        preprocessed_df = linear_regression_data_preprocessing()

        #train test split
        df_train, df_eval = preprocessed_df.randomSplit([0.8, 0.2], 42)

        #group into batches
        sorted_df = df_train.orderBy("Date")
        batches = sorted_df.groupBy("year", "month").agg(sql_fun.collect_list("features").alias("monthly features"), sql_fun.collect_list("AveragePrice").alias("monthly prices")) #now our batches consist of data generated over the period of each month

        #prepare LinearRegression Model and evaluator
        linRegr = LinearRegression(featuresCol="features", labelCol="AveragePrice")
        evaluator = RegressionEvaluator(labelCol="AveragePrice", predictionCol="prediction", metricName="rmse")

        cumulative_train_data = spark.createDataFrame([], df_train.select("features", "AveragePrice").schema) #we will store cumulative training results here

        for batch in batches.collect():
            year, month, monthly_features, monthly_prices = batch["year"], batch["month"], batch["monthly features"], batch["monthly prices"]

            #create batch dataframe
            batch_df = spark.createDataFrame(zip(monthly_features, monthly_prices), schema=["features", "AveragePrice"])

            cumulative_train_data = cumulative_train_data.union(batch_df)

            model = linRegr.fit(cumulative_train_data)
            
            #check the results for each iteration
            predictions = model.transform(df_eval)
            rmse = evaluator.evaluate(predictions)
            
            #view iteration predictions
            predictions.select("Date", "AveragePrice", "prediction").show(5, truncate=False)

            #view iteration rmse
            print(f"After training on {year}-{month}, RMSE on test set: {rmse}")

    elif (model == RegressionModels.RANDOM_FOREST):
        #Random Forest preprocess and training here
        pass
    
    elif (model == RegressionModels.XGBOOST):
        #XGboost here
        pass

    else:
        #test block here
        df.select("Date").groupBy("Date").count().orderBy("Date").show()



    # #COMBINING FEATURES

    # vect = VectorAssembler(inputCols=df.columns[1:], outputCol="features")
    # df_train = vect.transform(df_train)
    # df_eval = vect.transform(df_eval)

    # df_train = df_train.select("AveragePrice", "features")
    # df_train.show(10)

    # #FEATURE SCALING

    # scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    # scaler_model = scaler.fit(df_train)
    # df_train = scaler_model.transform(df_train)
    # df_eval = scaler_model.transform(df_eval)

    # df_train = df_train.select("AveragePrice", "features", "scaledFeatures")

    # df_train.show(10)

    # #

    # #RANDOM FOREST REGRESSOR MODEL

    # results = random_forest_regression_test(df_train, df_eval)

    # #XGBoost
    ...

    spark.sparkContext.stop()

    

