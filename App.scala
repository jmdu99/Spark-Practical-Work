import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}
import org.apache.spark.sql.functions.{when,lit,col,length,concat_ws,substring,avg,desc,date_format}
import org.apache.spark.ml.feature.{Imputer, StringIndexer,OneHotEncoder,VectorAssembler,Normalizer,UnivariateFeatureSelector}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.linalg.SparseVector


object App {
  
  /*
   The Spark Application has been executed in a Google Cloud cluster with 2 worker nodes
   (free period of 300$) because it takes so much time to execute in local
   
   Followed steps for running the app:
   			1. cd sparkapp -> root folder of the project
   			2. sbt package -> for compiling and generating the JAR file
   			3. gsutil cp target/scala-2.12/sparkapp_2.12-1.0.0.jar gs://aesthetic-kite-332712 -> copy local JAR into the project bucket
   			4. gcloud dataproc clusters start spark-cluster --region=us-central1  -> start the project cluster
   			5. gcloud dataproc jobs submit spark --region=us-central1 --cluster=spark-cluster --class=App --jars=gs://aesthetic-kite-332712/sparkapp_2.12-1.0.0.jar -- gs://aesthetic-kite-332712/2008.csv -> Execute the job
   			6. gcloud dataproc clusters stop spark-cluster --region=us-central1 -> stop the project cluster
   			
    However, it can be executed in local as well:
   		  spark-submit --master local --class App $JAR_PATH $CSV_PATH
   		  For example:
   		  	$JAR_PATH=target/scala-2.12/sparkapp_2.12-1.0.0.jar
   		  	$CSV_PATH="src/main/resources/2008.csv"
   	
   	Note:
   	    - The class (App.scala) is in the default package (put only App in --class)
   	    - $JAR_PATH and $CSV_PATH can be absolute paths as well (the JAR and the input CSV can be in any location)  	  	
  */
  
	def main(args : Array[String]) {
		Logger.getLogger("org").setLevel(Level.FATAL)
		val spark = SparkSession.builder().appName("Spark Practical Work")
		//.master("local")
		.getOrCreate()

		import spark.implicits._

		// After checking all CSV fields, we assign correct types
		val schema = new StructType()
		.add("Year",StringType,true)
		.add("Month",StringType,true)
		.add("DayofMonth",StringType,true)
		.add("DayOfWeek",StringType,true)
		.add("DepTime",StringType,true)
		.add("CRSDepTime",StringType,true)
		.add("ArrTime",StringType,true)
		.add("CRSArrTime",StringType,true)
		.add("UniqueCarrier",StringType,true)
		.add("FlightNum",StringType,true)
		.add("TailNum",StringType,true)
		.add("ActualElapsedTime",IntegerType,true)
		.add("CRSElapsedTime",IntegerType,true)
		.add("AirTime",IntegerType,true)
		.add("ArrDelay",IntegerType,true)
		.add("DepDelay",IntegerType,true)
		.add("Origin",StringType,true)
		.add("Dest",StringType,true)
		.add("Distance",IntegerType,true)
		.add("TaxiIn",IntegerType,true)
		.add("TaxiOut",IntegerType,true)
		.add("Cancelled",IntegerType,true)
		.add("CancellationCode",StringType,true)
		.add("Diverted",IntegerType,true)
		.add("CarrierDelay",IntegerType,true)
		.add("WeatherDelay",IntegerType,true)
		.add("NASDelay",IntegerType,true)
		.add("SecurityDelay",IntegerType,true)
		.add("LateAircraftDelay",IntegerType,true)
		
		// Some numeric columns such as DayofMonth have been considerated as string
		// because although the columns have numbers, they don't have a cuantitative value, 
		// that is, all the values must be compared in the same way (ex: day 4 is not greater than day 3)

		// CSV into Dataframe
		var df = spark.read.options(Map("header"->"true", "nanValue" -> "NA", "emptyValue" -> ""))
		.schema(schema)
		.csv(args(0)) //Example: "src/main/resources/2008.csv" as input parameter

		println("Read dataframe: ") 
		df.show(10,false)
		//-------------------INITIAL PREPROCESSING---------------------------------------------
		// Dataframe with forbidden variables removed
		df = df
		.drop("ArrTime")
		.drop("ActualElapsedTime")
		.drop("AirTime")
		.drop("TaxiIn")
		.drop("Diverted")
		.drop("CarrierDelay")
		.drop("WeatherDelay")
		.drop("NASDelay")
		.drop("SecurityDelay")
		.drop("LateAircraftDelay")

		// Since the objective is to predict the arrival time, we don't want those flights that have been cancelled (no arrival time)
		// So no cancelled flights are going to be filtered and Cancelled and CancellationCode columns will be removed
		// Also estimated elapsed time (CRSElapsedTime) can't be negative (filter positive values)
		// Duplicated flights are removed so that they do not have more importance
		df = df.filter("Cancelled == 0")
		.drop("CancellationCode")
		.drop("Cancelled")
		.filter("CRSElapsedTime > 0")
		.distinct() // 4 duplicated rows in 2008.csv

		//df.show(20)

		// Now the values of column DayOfWeek are going to be renamed/transformed for clarity purposes
		df = df.withColumn("DayOfWeek",
				when(col("DayOfWeek").rlike("1"),lit("Monday"))
				.when(col("DayOfWeek").rlike("2"),lit("Tuesday"))
				.when(col("DayOfWeek").rlike("3"),lit("Wednesday"))
				.when(col("DayOfWeek").rlike("4"),lit ("Thursday"))
				.when(col("DayOfWeek").rlike("5"),lit("Friday"))
				.when(col("DayOfWeek").rlike("6"),lit("Saturday"))
				.when(col("DayOfWeek").rlike("7"),lit("Sunday"))
				)

		//df.show(20)

		// Also we are going to create a derivated column Date 
		// instead of having year, months and days in separated columns 
		// because it makes more sense to consider full dates (e.g. 2019/06/24)
		// In order to do it, we are going to concat Year, Month and DayofMonth columns             

		df = df
		// Create Date column
		.withColumn("Date",concat_ws("/",col("Year"),col("Month"),col("DayofMonth")))
		// We don't cast the column to Date format since VectorAssembler only accepts numerical and strings features
		.drop("Month")
		.drop("DayofMonth")
		.drop("Year")
		
		//df.show(20)

		// Hours in proper format (hh:mm) for DepTime, CRSDepTime and CRSArrTime columns  
		// We are going to add a 0 at the beginning for those hours whose length is 3
		df = df
		.withColumn("DepTime",  when(length(col("DepTime")) < 4 , concat_ws("",lit("0"),col("DepTime"))).otherwise(col("DepTime")))
		.withColumn("CRSDepTime",  when(length(col("CRSDepTime")) < 4 , concat_ws("",lit("0"),col("CRSDepTime"))).otherwise(col("CRSDepTime")))
		.withColumn("CRSArrTime",  when(length(col("CRSArrTime")) < 4 , concat_ws("",lit("0"),col("CRSArrTime"))).otherwise(col("CRSArrTime")))
		
		// Also we are going to add ':' to all the hours for clarity purposes
		.withColumn("DepTime",  concat_ws(":",substring(col("DepTime"),1,2),substring(col("DepTime"),3,4)))
		.withColumn("CRSDepTime", concat_ws(":",substring(col("CRSDepTime"),1,2),substring(col("CRSDepTime"),3,4)))
		.withColumn("CRSArrTime", concat_ws(":",substring(col("CRSArrTime"),1,2),substring(col("CRSArrTime"),3,4)))
		
		// Once we have hours in correct format, we tranaform hours to day intervals (morning, afternoon, evening and night)
		// Maybe it's easier for the model to find patterns in this way
		df = df
		.withColumn("Time", date_format(col("DepTime"), "HH:mm"))
    .withColumn("DepTime", when(col("Time") > "06:00" && col("Time") <= "12:00", "Morning")
                           .when(col("Time") > "12:00" && col("Time") <= "17:00", "Afternoon")
                           .when(col("Time") > "17:00" && col("Time") <= "20:00", "Evening")
                           .otherwise("Night"))
    .withColumn("CRSDepTime", when(col("Time") > "06:00" && col("Time") <= "12:00", "Morning")
                           .when(col("Time") > "12:00" && col("Time") <= "17:00", "Afternoon")
                           .when(col("Time") > "17:00" && col("Time") <= "20:00", "Evening")
                           .otherwise("Night"))
    .withColumn("CRSArrTime", when(col("Time") > "06:00" && col("Time") <= "12:00", "Morning")
                           .when(col("Time") > "12:00" && col("Time") <= "17:00", "Afternoon")
                           .when(col("Time") > "17:00" && col("Time") <= "20:00", "Evening")
                           .otherwise("Night"))
    .drop("Time")

    df.select("DepTime","CRSDepTime","CRSArrTime").show(10,false)
		// Since DepTime, CRSDepTime and CRSArrTime have almost the same information, 
    // we are going to delete CRSDepTime and CRSArrTime columns
		df = df.drop("CRSDepTime").drop("CRSArrTime")
		
		println("Dataframe schema: ") 
		df.printSchema()
		println("Preprocessed dataframe: ") 
		df.show(10)
		println("Done")
		println("-------------------")

		//-------------------EXPLORATORY DATA ANALYSIS---------------------------------------------
		// Once we have the data in desired format, it is time to explore it:
		// find null / missing values, outliers, view some statistics...
		println("Some statistics: ")
		df.select("CRSElapsedTime","ArrDelay","DepDelay","Distance","TaxiOut").describe().show(false)
		println("Done")
		// It seems that there are some missing/null values in ArrDelay column -> replace null values
		// Variances are in different scales-> normalization/standardization of variables
		// Min and Max values seem to be OK in all columns
		
		// Correlations -> explanatory variables should not be correlated
		println("----------------------------------------------------------")
		println("Correlations of explanatory variables with target variable")
		println("----------------------------------------------------------")
		println("Correlation between DepDelay and ArrDelay:")
		println(df.stat.corr("DepDelay", "ArrDelay", "pearson"))
		println("Correlation between CRSElapsedTime and ArrDelay:")
		println(df.stat.corr("CRSElapsedTime", "ArrDelay", "pearson"))
		println("Correlation between Distance and ArrDelay:")
		println(df.stat.corr("Distance", "ArrDelay", "pearson"))
		println("Correlation between TaxiOut and ArrDelay:")
		println(df.stat.corr("TaxiOut", "ArrDelay", "pearson"))
		println("----------------------------------------------------------")
		println("Correlations between explanatory variables")
		println("----------------------------------------------------------")
		println("Correlation between CRSElapsedTime and DepDelay:")
		println(df.stat.corr("CRSElapsedTime", "DepDelay", "pearson"))
		println("Correlation between CRSElapsedTime and Distance:")
		println(df.stat.corr("CRSElapsedTime", "Distance", "pearson"))
		println("Correlation between CRSElapsedTime and TaxiOut:")
		println(df.stat.corr("CRSElapsedTime", "TaxiOut", "pearson"))
		println("Correlation between DepDelay and Distance:")
		println(df.stat.corr("DepDelay", "Distance", "pearson"))
		println("Correlation between DepDelay and TaxiOut:")
		println(df.stat.corr("DepDelay", "TaxiOut", "pearson"))
		println("Correlation between Distance and TaxiOut:")
		println(df.stat.corr("Distance", "TaxiOut", "pearson"))
		println("----------------------------------------------------------")
		// As expected, DepDelay and ArrDelay are highly correlated
		// CRSElapsedTime is almost not related with ArrDelay
		// CRSElapsedTime is highly related to Distance -> CRSElapsedTime will be deleted (or Distance could be deleted as well)
		df = df.drop("CRSElapsedTime")

		//-------------------FIND & REPLACE MISSING VALUES-----------------------------------------   
		// Replace null and missing values, we check all columns in case a different dataset is used
		println("Replacing null and missing values with each column mean (numeric columns) or mode (string columns)")
		// Replace null / missing in numeric columns by column mean
		val numeric_columns = Array("ArrDelay","DepDelay","TaxiOut","Distance")
		val imputer = new Imputer()
		.setInputCols(numeric_columns)
		.setOutputCols(numeric_columns)
		.setStrategy("mean")
		df = imputer.fit(df).transform(df)

		val string_columns = df.columns.diff(numeric_columns)
		// Replace null / missing in string columns by column mode
		string_columns.map(column =>{
			var mode = df.groupBy(column).count().orderBy(desc(column)).first()(0).toString()
					df = df.na.fill(mode,Array(column))
		})
		println("Done")
		println("-------------------")
		/*
    // Check if it works
    df.columns.map(column => println(df.filter(col(column).isNull || col(column).isNaN).count()))
    df.show(10)
		 */
		
		// Find outliers
		println("Finding outliers")
		numeric_columns.map(col => {
		val quantiles = df.stat.approxQuantile(s"$col", Array(0.25,0.75),0.0)
    val Q1 = quantiles(0)
    val Q3 = quantiles(1)
    val IQR = Q3 - Q1
    val lowerRange = Q1 - 1.5*IQR
    val upperRange = Q3+ 1.5*IQR

    val outliers = df.filter(s"$col < $lowerRange or $col > $upperRange").select(col)
    println(s"Outliers in $col column: ")
    outliers.show(10)
    println(s" Number of outliers in $col: ${outliers.count}")
		})
		println("Done")
		println("-------------------")
		// We can see delays of 40 or 50 minutes, and in some cases hours
		// Also huge distance in some flights
		// However, these situations happen often, so they need to be taken into account in the models, 
		// Hence we are not going to delete any outlier
		
		println("Dataframe schema: ") 
		df.printSchema()
		println("Dataframe after exploratory data analysis: ") 
		df.show(10,false)

		//-------------------VECTOR OF FEATURES & NORMALIZATION-----------------------------------------
		// Before getting the vector of features and normalizing, we need to encode categorical features
		// StringIndexer -> OneHotEncoder -> VectorAssembler -> Normalizer  
		println("Getting vector of normalized features")
		val index_columns = string_columns.map(col => col + "Index")

		// StringIndexer
		val indexer = new StringIndexer()
		.setInputCols(string_columns)
		.setOutputCols(index_columns)

		val vec_columns = string_columns.map(col => col + "Vec")

		// OneHotEncoder
		val encoder = new OneHotEncoder()
		.setInputCols(index_columns)
		.setOutputCols(vec_columns)

		// VectorAssembler
		val num_vec_columns:Array[String] = (numeric_columns.filter(!_.contains("ArrDelay"))) ++ vec_columns   
		val assembler = new VectorAssembler()
		.setInputCols(num_vec_columns)
		.setOutputCol("features")

		// Normalizer
		val normalizer = new Normalizer()
		.setInputCol("features")
		.setOutputCol("normFeatures")
		.setP(1.0)

		// All together in pipeline
		val pipeline = new Pipeline()
		.setStages(Array(indexer, encoder, assembler,normalizer))
		df = pipeline.fit(df).transform(df)
		df.printSchema()
		df.show(10,false)
		println("Done")
		println("-------------------")

		//-------------------FSS-----------------------------------------
		// Models have not been fitted without FSS due to the huge number of features (large time of execution)
    
    // In order to configurate the models, 3-fold cross validation has been applied
    // After getting the best model for all the algorithms, the models have been evaluated
    // with a test set for each FSS
    
    // Selectors have not been considerated for cross validation since we wanted 
    // to see and compare the performance for all the different FSS
		
		val selector_fpr = new UnivariateFeatureSelector()
         .setFeatureType("continuous")
         .setLabelType("continuous") // score function -> F-value (f_regression)
         .setSelectionThreshold(0.05)
         .setSelectionMode("fpr") // false positive rate
         .setFeaturesCol("normFeatures")
         .setLabelCol("ArrDelay")
         .setOutputCol("selectedFeatures")
         
   val selector_fdr = new UnivariateFeatureSelector()
         .setFeatureType("continuous")
         .setLabelType("continuous") // score function -> F-value (f_regression)
         .setSelectionThreshold(0.05)
         .setSelectionMode("fdr") // false discovery rate
         .setFeaturesCol("normFeatures")
         .setLabelCol("ArrDelay")
         .setOutputCol("selectedFeatures")
         
    val selector_fwe = new UnivariateFeatureSelector()
         .setFeatureType("continuous")
         .setLabelType("continuous") // score function -> F-value (f_regression)
         .setSelectionThreshold(0.05)
         .setSelectionMode("fwe") // family-wise error rate
         .setFeaturesCol("normFeatures")
         .setLabelCol("ArrDelay")
         .setOutputCol("selectedFeatures")
    
    val row = df.select("normFeatures").head
    val vector = row(0).asInstanceOf[SparseVector]
		println(s"Number of features without FSS: ${vector.size}")
   
    println("Performing FSS selection - false positive rate")
    val fpr = selector_fpr.fit(df)
    val df_fpr = fpr.transform(df)
    println("Done")
    println(s"Number of features after applying false positive rate FSS: ${fpr.selectedFeatures.length}")
    println("Performing FSS selection - false discovery rate")
    val fdr = selector_fdr.fit(df)
    val df_fdr = fdr.transform(df)
    println("Done")
    println(s"Number of features after applying false discovery rate FSS: ${fdr.selectedFeatures.length}")
    println("Performing FSS selection - family-wise error rate")
    val fwe = selector_fwe.fit(df)
    val df_fwe = fwe.transform(df)
    println("Done")
    println(s"Number of features after applying family-wise error FSS: ${fwe.selectedFeatures.length}")
         
    val Array(trainingData_fpr, testData_fpr) = df_fpr.randomSplit(Array(0.7, 0.3),10)
    val Array(trainingData_fdr, testData_fdr) = df_fdr.randomSplit(Array(0.7, 0.3),10)
    val Array(trainingData_fwe, testData_fwe) = df_fwe.randomSplit(Array(0.7, 0.3),10)
		
    /*
		 As it is a linear regression problem, that is, explaining or predicting the value of a continuous variable, 
		 we have tried to apply many algorithms within the family of regression algorithms 
		 (also trying not to extend the execution time more than 24 hours in a row ) and then compare the 
		 different results obtained. Specifically, the following algorithms have been applied: Linear regression, 
		 Generalized linear regression, Decision tree regression and Random forest regression.
		 */
		
		//-------------------LinearRegression-----------------------------------------
		val lr = new LinearRegression()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("selectedFeatures")
				.setPredictionCol("predictionLR")
				.setMaxIter(10)

		val lr_paramGrid = new ParamGridBuilder()
				.addGrid(lr.regParam, Array(0.1, 0.01))
				.addGrid(lr.elasticNetParam, Array(1, 0.8, 0.5))
				.build()

		val lr_evaluator_rmse = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionLR")
				.setMetricName("rmse")

		val lr_evaluator_r2 = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionLR")
				.setMetricName("r2")

		val lr_cv = new CrossValidator()
				.setEstimator(lr)
				.setEvaluator(lr_evaluator_rmse)
				.setEstimatorParamMaps(lr_paramGrid)
				.setNumFolds(3)
				.setParallelism(3)

		println("-------------------------Linear Regression FPR-------------------------")
		val lr_model_fpr = lr_cv.fit(trainingData_fpr)
		println("Model parameters:")
		println(lr_model_fpr.bestModel.extractParamMap())
		val lr_predictions_fpr = lr_model_fpr.transform(testData_fpr)
		println("ArrDelay VS predictionLR:")
		lr_predictions_fpr.select("ArrDelay", "predictionLR").show(10,false)
		println(s"Root Mean Squared Error = ${lr_evaluator_rmse.evaluate(lr_predictions_fpr)}")
		println(s"R-Squared = ${lr_evaluator_r2.evaluate(lr_predictions_fpr)}")
		
	 	println("-------------------------Linear Regression FDR-------------------------")
		val lr_model_fdr = lr_cv.fit(trainingData_fdr)
		println("Model parameters:")
		println(lr_model_fdr.bestModel.extractParamMap())
		val lr_predictions_fdr = lr_model_fdr.transform(testData_fdr)
		println("ArrDelay VS predictionLR:")
		lr_predictions_fdr.select("ArrDelay", "predictionLR").show(10,false)
		println(s"Root Mean Squared Error = ${lr_evaluator_rmse.evaluate(lr_predictions_fdr)}")
		println(s"R-Squared = ${lr_evaluator_r2.evaluate(lr_predictions_fdr)}")
		
		println("-------------------------Linear Regression FWE-------------------------")
		val lr_model_fwe = lr_cv.fit(trainingData_fwe)
		println("Model parameters:")
		println(lr_model_fwe.bestModel.extractParamMap())
		val lr_predictions_fwe = lr_model_fwe.transform(testData_fwe)
		println("ArrDelay VS predictionLR:")
		lr_predictions_fwe.select("ArrDelay", "predictionLR").show(10,false)
		println(s"Root Mean Squared Error = ${lr_evaluator_rmse.evaluate(lr_predictions_fwe)}")
		println(s"R-Squared = ${lr_evaluator_r2.evaluate(lr_predictions_fwe)}")

		//-------------------GeneralizedLinearRegression-----------------------------------------
		val glr = new GeneralizedLinearRegression()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("selectedFeatures")
				.setPredictionCol("predictionGLR")
				.setLink("identity")
				.setFamily("gaussian")
				.setMaxIter(10)

		val glr_paramGrid = new ParamGridBuilder()
				.addGrid(glr.regParam, Array(0.1, 0.01))
				.build()

		val glr_evaluator_rmse = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionGLR")
				.setMetricName("rmse")

		val glr_evaluator_r2 = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionGLR")
				.setMetricName("r2")

		val glr_cv = new CrossValidator()
				.setEstimator(glr)
				.setEvaluator(glr_evaluator_rmse)
				.setEstimatorParamMaps(glr_paramGrid)
				.setNumFolds(3) 
				.setParallelism(3)
				
	  // No False Positive Rate and False Discovery Rate FSS for Generalized Linear Regression
		// since the number of features has to be <= 4096
		
		println("-------------------------Generalized Linear Regression FWE-------------------------")
		val glr_model_fwe = glr_cv.fit(trainingData_fwe)
	  println("Model parameters:")
		println(glr_model_fwe.bestModel.extractParamMap())
		val glr_predictions_fwe = glr_model_fwe.transform(testData_fwe)
		println("ArrDelay VS predictionGLR:")
		glr_predictions_fwe.select("ArrDelay", "predictionGLR").show(10,false)
		println(s"Root Mean Squared Error = ${glr_evaluator_rmse.evaluate(glr_predictions_fwe)}")
		println(s"R-Squared = ${glr_evaluator_r2.evaluate(glr_predictions_fwe)}")
		
		// No parameters selected in Cross Validation for DecisionTreeRegression and RandomForestRegression
		// due to the large time of execution (even using a Google Cloud spark cluster)
		
		// We also tried models with Gradient-Boosted Tree Regression algorithm but the time of execution was excesive
		// and we were unable to get any result
		
		//-------------------DecisionTreeRegression-----------------------------------------
		val dtr = new DecisionTreeRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("selectedFeatures")
				.setPredictionCol("predictionDTR")

		val dtr_evaluator_rmse = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionDTR")
				.setMetricName("rmse")

		val dtr_evaluator_r2 = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionDTR")
				.setMetricName("r2")

		val dtr_cv = new CrossValidator()
				.setEstimator(dtr)
				.setEvaluator(dtr_evaluator_rmse)
				.setEstimatorParamMaps(new ParamGridBuilder().build())
				.setNumFolds(3) 
				.setParallelism(3)

		println("-------------------------Decision Tree Regression FPR-------------------------")
		val dtr_model_fpr = dtr_cv.fit(trainingData_fpr)
		println("Model parameters:")
		println(dtr_model_fpr.bestModel.extractParamMap())
		val dtr_predictions_fpr = dtr_model_fpr.transform(testData_fpr)
		println("ArrDelay VS predictionDTR:")
		dtr_predictions_fpr.select("ArrDelay", "predictionDTR").show(10,false)
		println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fpr)}")
		println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fpr)}")
		
		println("-------------------------Decision Tree Regression FDR-------------------------")
		val dtr_model_fdr = dtr_cv.fit(trainingData_fdr)
		println("Model parameters:")
		println(dtr_model_fdr.bestModel.extractParamMap())
		val dtr_predictions_fdr = dtr_model_fdr.transform(testData_fdr)
		println("ArrDelay VS predictionDTR:")
		dtr_predictions_fdr.select("ArrDelay", "predictionDTR").show(10,false)
		println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fdr)}")
		println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fdr)}")
		
		println("-------------------------Decision Tree Regression FWE-------------------------")
		val dtr_model_fwe = dtr_cv.fit(trainingData_fwe)
		println("Model parameters:")
		println(dtr_model_fwe.bestModel.extractParamMap())
		val dtr_predictions_fwe = dtr_model_fwe.transform(testData_fwe)
		println("ArrDelay VS predictionDTR:")
		dtr_predictions_fwe.select("ArrDelay", "predictionDTR").show(10,false)
		println(s"Root Mean Squared Error = ${dtr_evaluator_rmse.evaluate(dtr_predictions_fwe)}")
		println(s"R-Squared = ${dtr_evaluator_r2.evaluate(dtr_predictions_fwe)}")

		//-------------------RandomForestRegression-----------------------------------------
		val rfr = new RandomForestRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("selectedFeatures")
				.setPredictionCol("predictionRFR")

		val rfr_evaluator_rmse = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionRFR")
				.setMetricName("rmse")

		val rfr_evaluator_r2 = new RegressionEvaluator()
				.setLabelCol("ArrDelay")
				.setPredictionCol("predictionRFR")
				.setMetricName("r2")

		val rfr_cv = new CrossValidator()
				.setEstimator(rfr)
				.setEvaluator(rfr_evaluator_rmse)
				.setEstimatorParamMaps(new ParamGridBuilder().build())
				.setNumFolds(3) 
				.setParallelism(3)
				
		println("------------------------Random Forest Regression FPR-------------------------")
		val rfr_model_fpr = rfr_cv.fit(trainingData_fpr)
		println("Model parameters:")
		println(rfr_model_fpr.bestModel.extractParamMap())
		val rfr_predictions_fpr = rfr_model_fpr.transform(testData_fpr)
		println("ArrDelay VS predictionRFR:")
		rfr_predictions_fpr.select("selectedFeatures", "ArrDelay", "predictionRFR").show(10,false)
		println(s"Root Mean Squared Error = ${rfr_evaluator_rmse.evaluate(rfr_predictions_fpr)}")
		println(s"R-Squared = ${rfr_evaluator_r2.evaluate(rfr_predictions_fpr)}")
		
		println("------------------------Random Forest Regression FDR-------------------------")
		val rfr_model_fdr = rfr_cv.fit(trainingData_fdr)
		println("Model parameters:")
		println(rfr_model_fdr.bestModel.extractParamMap())
		val rfr_predictions_fdr = rfr_model_fdr.transform(testData_fdr)
		println("ArrDelay VS predictionRFR:")
		rfr_predictions_fdr.select("selectedFeatures", "ArrDelay", "predictionRFR").show(10,false)
		println(s"Root Mean Squared Error = ${rfr_evaluator_rmse.evaluate(rfr_predictions_fdr)}")
		println(s"R-Squared = ${rfr_evaluator_r2.evaluate(rfr_predictions_fdr)}")
		
		println("------------------------Random Forest Regression FWE-------------------------")
		val rfr_model_fwe = rfr_cv.fit(trainingData_fwe)
		println("Model parameters:")
		println(rfr_model_fwe.bestModel.extractParamMap())
		val rfr_predictions_fwe = rfr_model_fwe.transform(testData_fwe)
		println("ArrDelay VS predictionRFR:")
		rfr_predictions_fwe.select("selectedFeatures", "ArrDelay", "predictionRFR").show(10,false)
		println(s"Root Mean Squared Error = ${rfr_evaluator_rmse.evaluate(rfr_predictions_fwe)}")
		println(s"R-Squared = ${rfr_evaluator_r2.evaluate(rfr_predictions_fwe)}")

		// Summary table with RMSE and R2 measures for all the trained models
		// R2 measures the variability of the dependent variable (ArrDelay) that is explained by the predictors (must be independant variables)
		// RMSE is the square root of the mean of the square of all of the error (residuals)
		println("------------------------Summary-------------------------")
		val summaryDF = Seq(
						("LINEAR REGRESSION - False Positive Rate Selection",lr_evaluator_rmse.evaluate(lr_predictions_fpr), lr_evaluator_r2.evaluate(lr_predictions_fpr)),
						("LINEAR REGRESSION - False Discovery Rate Selection",lr_evaluator_rmse.evaluate(lr_predictions_fdr), lr_evaluator_r2.evaluate(lr_predictions_fdr)),
						("LINEAR REGRESSION - Family-Wise Error Rate Selection",lr_evaluator_rmse.evaluate(lr_predictions_fwe), lr_evaluator_r2.evaluate(lr_predictions_fwe)),
						("GENERALIZED LINEAR REGRESSION - Family-Wise Error Rate Selection",glr_evaluator_rmse.evaluate(glr_predictions_fwe), glr_evaluator_r2.evaluate(glr_predictions_fwe)),
						("DECISION TREE REGRESSION - False Positive Rate Selection",dtr_evaluator_rmse.evaluate(dtr_predictions_fpr),dtr_evaluator_r2.evaluate(dtr_predictions_fpr)),
						("DECISION TREE REGRESSION - False Discovery Rate Selection",dtr_evaluator_rmse.evaluate(dtr_predictions_fdr),dtr_evaluator_r2.evaluate(dtr_predictions_fdr)),
						("DECISION TREE REGRESSION - Family-Wise Error Rate Selection",dtr_evaluator_rmse.evaluate(dtr_predictions_fwe),dtr_evaluator_r2.evaluate(dtr_predictions_fwe)),
						("RANDOM FOREST REGRESSION - False Positive Rate Selection",rfr_evaluator_rmse.evaluate(rfr_predictions_fpr), rfr_evaluator_r2.evaluate(rfr_predictions_fpr)),
						("RANDOM FOREST REGRESSION - False Discovery Rate Selection",rfr_evaluator_rmse.evaluate(rfr_predictions_fdr), rfr_evaluator_r2.evaluate(rfr_predictions_fdr)),
						("RANDOM FOREST REGRESSION - Family-Wise Error Rate Selection",rfr_evaluator_rmse.evaluate(rfr_predictions_fwe), rfr_evaluator_r2.evaluate(rfr_predictions_fwe)))
				.toDF("Algorithm","RMSE","R2")
				
	 summaryDF.show(false)
	 
	 /*
	  The results obtained in the 10 trained models are quite similar in both RMSE and R2. 
	  On average, the RMSE is around 23.34 and the R2 is around 0.65, that is, 
	  65% of the variability of the dependent variable can be explained by the predictor variables (independent).

    It is observed that none of the 3 univariate filtering selection modes applied to the column of normalized 
    features produce models that make a difference in excessive performance over the rest. Therefore, it could be thought 
    that none of the variables considered to be in the regression models after the EDA (before vectorizing) 
    interfere or generate excessive noise when explaining or predicting the class variable.

    In future studies, it would be interesting to verify this idea using the entire column of normalized features 
    without applying univariate filtering, that is, without eliminating any features, and comparing the results obtained. 
    In order to successfully execute all the regression algorithms, a more powerful cluster (with more worker nodes 
    for example) would be needed since the original normalized features column has 13020 features. 
	  */
	}
}
