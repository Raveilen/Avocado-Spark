#imports
from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_fun;
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from enum import Enum

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from xgboost import XGBRegressor
from xgboost.spark import SparkXGBRegressor

class RegressionModels(Enum):
    LINEAR_REGRESSION = 1
    RANDOM_FOREST = 2
    XGBOOST = 3


#core variables
spark = SparkSession.builder.appName("avocado_price_prediction").getOrCreate()
df = spark.read.csv("Augmented_avocado.csv", sep = ",", header=True, inferSchema = True).drop("Unnamed: 0") #Dataset Load from file https://www.kaggle.com/datasets/mathurinache/avocado-augmented
numerical_features = ["Total Volume", "4046", "4225", "4770", "Total Bags", "year", "month"] #numerical features we consider in model training

input_features = list(numerical_features) #numerical and categorical features we consider in model training (categorical should be converted to numerical in preprocesing phase)
input_features.append("type_encoded")

model = RegressionModels.XGBOOST #we choose regression method here


#functions
def IQR(): #Interquantile Range - outlier removal method. It removes values which are beyond specified range

    quantiles = df.approxQuantile("AveragePrice", [0.25, 0.75], 0.05)
    Q1 = quantiles[0]
    Q3 = quantiles[1]

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_df = df.filter((sql_fun.col("AveragePrice") >= lower_limit) & (sql_fun.col("AveragePrice") <= upper_limit))

    return filtered_df

def replace_date_with_seasons(df): #generates additional information about seasons (not in use at this moment)

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

def linear_regression_preprocessing(): #preprocesing for Linear Regression (includes Standard Scaler usage). Returns preprocessed dataset

    #scale numerical features
    numerical_assembler = VectorAssembler(inputCols=numerical_features, outputCol="numerical_features")
    scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features", withMean=True, withStd=True)

    #create features vector 
    final_assembler = VectorAssembler(inputCols=["scaled_numerical_features", "type_encoded"], outputCol="features")

    #create processing pipeline
    pipeline = Pipeline(stages=[numerical_assembler, scaler, indexer, encoder, final_assembler])

    preprocessed_df = pipeline.fit(df).transform(df)
    
    return preprocessed_df

def random_forest_and_xgb_processing(): #Like above but we skipped here Standard Scaler because it is not needed in RandomForest and XGBoost
    #create features vector 
    assemlber = VectorAssembler(inputCols=input_features, outputCol="features")

    pipeline = Pipeline(stages=[indexer, encoder, assemlber])

    preprocessed_df = pipeline.fit(df).transform(df)
    
    return preprocessed_df


if __name__ == '__main__' : 

    #Add month and year columns to dataset
    df = df.withColumn("year", sql_fun.year("Date")).withColumn("month", sql_fun.month("Date"))

    #remove outliers
    df = IQR()

    #encode categorical features
    indexer = StringIndexer(inputCol="type", outputCol="type_index")
    encoder = OneHotEncoder(inputCol="type_index", outputCol="type_encoded")

    #Based on selected RegresionModel method, preprocessing might be different (different estimator selected)
    if(model == RegressionModels.LINEAR_REGRESSION):
        preprocessed_df = linear_regression_preprocessing()
        estimator = LinearRegression(featuresCol="features", labelCol="AveragePrice")

    elif (model == RegressionModels.RANDOM_FOREST):
        preprocessed_df = random_forest_and_xgb_processing()
        estimator = RandomForestRegressor(featuresCol="features", labelCol="AveragePrice", numTrees=100)

    elif (model == RegressionModels.XGBOOST):
        preprocessed_df = random_forest_and_xgb_processing()
        estimator = SparkXGBRegressor(features_col="features", label_col="AveragePrice", prediction_col="prediction", eta=0.1, objective="reg:squarederror")
    else:
        #test block
        df.select("Date").groupBy("Date").count().orderBy("Date").show()
        exit()

    #train test split
    df_train, df_eval = preprocessed_df.randomSplit([0.8, 0.2], 42)

    #we need to sort both datasets chronoligicaly
    sorted_train = df_train.orderBy("Date")
    sorted_eval = df_eval.orderBy("Date")

    #then we extract available month-year pairs based on train dataset (model learns based on those dates so those points should be verified with test data)
    year_month_pairs = sorted_train.select("year", "month").distinct().orderBy("year", "month").collect()

    evaluator = RegressionEvaluator(labelCol="AveragePrice", predictionCol="prediction", metricName="rmse")

    #enumerate based on gathered month-year pairs
    for idx, row in enumerate(year_month_pairs[:-1]):  # Exclude the last pair
        
        #year, month for traing batch
        year, month = row["year"], row["month"]

        #next_year, next month for evaluation batch
        next_year, next_month = year_month_pairs[idx + 1]["year"], year_month_pairs[idx + 1]["month"]

        # Train on the current month
        train_batch = sorted_train.filter((sorted_train["year"] == year) & (sorted_train["month"] == month))
        model = estimator.fit(train_batch)

        # Test on the next month
        test_batch = sorted_eval.filter((sorted_eval["year"] == next_year) & (sorted_eval["month"] == next_month))

        if test_batch.count() > 0:  # Ensure there's test data for the next month
            predictions = model.transform(test_batch)

            rmse = evaluator.evaluate(predictions)
            
            #view representative samples
            # predictions.select("Date", "AveragePrice", "prediction").show(5, truncate=False)
            
            #view avg - returns small dataset representing expected and achieved avg prices for days in month when samples was taken.
            predictions_avg = predictions.select("Date", "AveragePrice", "prediction").groupBy("Date").agg(sql_fun.avg("AveragePrice").alias("avgAveragePrice"), sql_fun.avg("prediction").alias("avg_prediction"))
            predictions_avg.show(truncate=False)

            #view iteration rmse - value representing percent deviation between expected and result value [0-100] 10-30 is aceptable. Better visible in representative samples rather than avg values.
            print(f"After training on {year}-{month}, RMSE on test set: {rmse}")

    spark.sparkContext.stop()

    

