import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, month, round, avg, unix_timestamp, to_date
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator

def run_pipeline():
    print("Starting PySpark Pipeline with Project-Specific Features...")
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("MicroplasticPrediction") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Ingestion
    print("Loading data...")
    dataset_path = "ocean_sensor_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    df = spark.read.csv(dataset_path, header=True, inferSchema=True)
    
    # Fill any null values with 0
    df = df.na.fill(0)

    # Define Target: Hotspot if Microplastic_Concentration is in top 25% (using approxQuantile)
    quantiles = df.approxQuantile("Microplastic_Concentration", [0.75], 0.01)
    if not quantiles:
        threshold = df.approxQuantile("Microplastic_Concentration", [0.5], 0.01)[0]
    else:
        threshold = quantiles[0]
        
    df = df.withColumn("Hotspot", when(col("Microplastic_Concentration") > threshold, 1).otherwise(0))

    # 2. Advanced Feature Engineering (Spatio-Temporal)
    print("Performing Spatio-Temporal Feature Engineering...")
    # Create grid cells (rounded Lat/Lon) for spatial partitioning
    df = df.withColumn("Grid_Lat", round(col("Latitude"), 0))
    df = df.withColumn("Grid_Lon", round(col("Longitude"), 0))
    
    # Extract Seasonal Index
    df = df.withColumn("Date", to_date(col("Date")))
    df = df.withColumn("Seasonal_Index", month(col("Date")))
    
    # 30-day Rolling Window for Ocean Current Velocity (30 days = 30 * 86400 seconds)
    days = lambda i: i * 86400
    windowSpec = Window.partitionBy("Grid_Lat", "Grid_Lon").orderBy(unix_timestamp("Date").cast("long")).rangeBetween(-days(30), 0)
    df = df.withColumn("Rolling_30D_Current", avg("Ocean_Current_Velocity").over(windowSpec))
    
    # Fill any nulls that might have been created
    df = df.fillna(0)

    # Assembler for clustering
    features = [
        "Sea_Surface_Temperature", 
        "Salinity", 
        "Wind_Speed", 
        "Ocean_Current_Velocity", 
        "Chlorophyll_Concentration", 
        "Wave_Height",
        "Distance_from_Coastline",
        "Distance_from_River_Mouth",
        "Seasonal_Index",
        "Rolling_30D_Current"
    ]
    assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
    df_assembled = assembler.transform(df)

    # StandardScaler 
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    # 3. KMeans Clustering
    print("Running KMeans Clustering...")
    kmeans = KMeans().setK(5).setSeed(42).setFeaturesCol("features").setPredictionCol("cluster")
    kmeans_model = kmeans.fit(df_scaled)
    df_clustered = kmeans_model.transform(df_scaled)

    evaluator = ClusteringEvaluator(predictionCol="cluster")
    silhouette = evaluator.evaluate(df_clustered)
    print(f"Silhouette with squared euclidean distance = {silhouette}")

    # 4. RandomForest Classification
    print("Running RandomForest Classification...")
    # Using 'cluster' as an additional feature alongside our actual features
    rf_features = features + ["cluster"]
    assembler_rf = VectorAssembler(inputCols=rf_features, outputCol="rf_features_vec")
    df_rf_input = assembler_rf.transform(df_clustered)

    rf = RandomForestClassifier(labelCol="Hotspot", featuresCol="rf_features_vec", numTrees=30, maxDepth=10, seed=42)
    rf_model = rf.fit(df_rf_input)
    predictions = rf_model.transform(df_rf_input)

    # Evaluate RF
    evaluator_rf = MulticlassClassificationEvaluator(labelCol="Hotspot", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_rf.evaluate(predictions)
    print(f"RandomForest Accuracy: {accuracy}")

    # Feature Importance
    importances = rf_model.featureImportances
    feature_importance_dict = {rf_features[i]: float(importances[i]) for i in range(len(rf_features))}

    # 5. Export subset for Dashboard UI
    print("Exporting data to JSON for Dashboard...")
    # Using 2,000 points to keep frontend Folium snappy
    output_df = predictions.select("Latitude", "Longitude", "Sea_Surface_Temperature", "Salinity", "Wind_Speed", "Ocean_Current_Velocity", "Microplastic_Concentration", "cluster", "prediction", "Hotspot")
    pandas_df = output_df.sample(fraction=0.1, seed=42).limit(2000).toPandas()
    
    output_dir = "dashboard_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_json = pandas_df.to_dict(orient="records")
    
    # Save to file
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump({
            "metrics": {
                "silhouette_score": silhouette,
                "rf_accuracy": accuracy,
                "hotspot_threshold": threshold,
                "feature_importances": feature_importance_dict
            },
            "points": data_json
        }, f)
        
    print("Pipeline completed successfully!")
    spark.stop()

if __name__ == "__main__":
    run_pipeline()
