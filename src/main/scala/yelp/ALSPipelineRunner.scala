package yelp

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, _}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, explode, expr}
import yelp.ALSRunner.indexer

import scala.util.Random


object ALSPipelineRunner {
  def main(args: Array[String]) = {
    val spark = SparkSession.builder()
      .appName("Yelp reco")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // retrieve userId, businessId and ratings from raw data set. Using 0.5 sampling due to test on local
    val rawReviewDF = spark.read.format("json").load("yelp-dataset/yelp_academic_dataset_review.json")
      .sample(0.5)

    // index user_id and businessId to numeric type
    val indexedReviewDF = indexString(rawReviewDF)

    val Array(train, test) = indexedReviewDF.randomSplit(Array(0.7, 0.3))

    val als = new ALS()
      .setUserCol("user_no")
      .setItemCol("business_no")
      .setRatingCol("stars")
      .setPredictionCol("prediction")
      .setColdStartStrategy("drop")
      .setSeed(Random.nextLong())
      .setImplicitPrefs(false)
      .setNonnegative(true)
      .setNumUserBlocks(1) // default 10
      .setNumUserBlocks(1)

    val pipeline = new Pipeline()
      .setStages(Array(als))

    val params = new ParamGridBuilder()
      .addGrid(als.rank, Array(1, 2)) // default 10
      .addGrid(als.regParam, Array(0.01, 0.02)) // default 0.1
      .addGrid(als.alpha, Array(1.0, 2.0)) // default 10
      .addGrid(als.maxIter, Array(1))
      .build

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("stars")
      .setPredictionCol("prediction")

    val tsv = new TrainValidationSplit()
      .setTrainRatio(0.75)
      .setEstimatorParamMaps(params)
      .setEstimator(pipeline)
      .setEvaluator(evaluator)

    val tvsFitted = tsv.fit(train)

    val predictions = tvsFitted.transform(test)

    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse") // Root-mean-square error = 5.278573251959344

    // ranking evaluation
    val ranks = getRankingMetric(predictions, spark)
    println(s"Mean average precision = ${ranks.meanAveragePrecision}") // Mean average precision = 0.9853041114509005
    println(s"Prediction at 3 ${ranks.precisionAt(3)}") // Prediction at 3 = 0.2246687054026504
    /**
     * Root-mean-square error = 4.454759293413745
     * Mean average precision = 0.9024417718508368
     * Prediction at 3 0.5144786294707111
     */

    // print results
    val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
    val trainedALS = trainedPipeline.stages(0).asInstanceOf[ALSModel]

    // Generate top 3 business recommendations for a specified set of users
    val users = indexedReviewDF.select(trainedALS.getUserCol).distinct().limit(3)
    val userSubsetRecs = trainedALS.recommendForUserSubset(users, 3)
    userSubsetRecs.show(false)
    /**
     * +-------+---------------------------------------------------------------+
     * |user_no|recommendations                                                |
     * +-------+---------------------------------------------------------------+
     * |444410 |[[207801, 143.54684], [195392, 143.28445], [179736, 141.68715]]|
     * |347323 |[[207659, 336.00647], [209305, 333.65787], [131683, 330.3789]] |
     * |203854 |[[201256, 315.79272], [206973, 313.43314], [195286, 312.62524]]|
     * +-------+---------------------------------------------------------------+
     */


    val mergedDF = addUserAndBusinessNames(indexedReviewDF, userSubsetRecs, spark);
    mergedDF.show(false)

    /**
     * +-------+------+-------------------+----------------------------------+
     * |user_no|name  |reco               |name                              |
     * +-------+------+-------------------+----------------------------------+
     * |203854 |Lauren|[195286, 312.62524]|On Q Wax & Hair Studio            |
     * |203854 |Lauren|[206973, 313.43314]|Carter Electrical                 |
     * |203854 |Lauren|[201256, 315.79272]|Anew Medspa                       |
     * |347323 |Diana |[131683, 330.3789] |The Art Department                |
     * |347323 |Diana |[209305, 333.65787]|Hometown Heroes                   |
     * |347323 |Diana |[207659, 336.00647]|A-1 Auto Service                  |
     * |444410 |Kevin |[179736, 141.68715]|Breakthrough Performance and Rehab|
     * |444410 |Kevin |[195392, 143.28445]|Life Storage                      |
     * |444410 |Kevin |[207801, 143.54684]|Toy Florist                       |
     * +-------+------+-------------------+----------------------------------+
     */

  }

  /**
   * indexing user_id and business_id for ALS model
   * @param rawReviewDF which is read from file
   * @return indexed Dataframe
   */
  def indexString(rawReviewDF: Dataset[Row]) = {
    val reviewDF = rawReviewDF
      .select(col("user_id"), col("business_id"), col("stars"))

    // indexing userId and business for ALS
    val userIndexedReviewDF = indexer("user_id", "user_no", reviewDF)
    val businessIndexedReviewDF = indexer("business_id", "business_no", userIndexedReviewDF)
    businessIndexedReviewDF
  }

  import org.apache.spark.ml.feature.StringIndexer

  def indexer(inputCol: String, outputCol: String, df: DataFrame) = new StringIndexer()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
    .fit(df)
    .transform(df)

  /**
   * Get ranking metrics of actual user ranking and predicted raking
   * @param predictions result of prediction from model
   * @param spark spark session
   * @return ranking metric
   */
  def getRankingMetric(predictions: DataFrame, spark: SparkSession): RankingMetrics[Double] = {
    val userActual = predictions
      .where("stars > 2")
      .groupBy("user_no")
      .agg(expr("collect_set(business_no) as business"))

    val userPrediction = predictions
      .orderBy(col("user_no"), col("prediction").desc)
      .groupBy("user_no")
      .agg(expr("collect_set(business_no) as business"))

    import spark.implicits._
    val a = userActual
      .join(userPrediction, Seq("user_no"))
      .map(row => (row(1).asInstanceOf[Seq[Double]].toArray, row(2).asInstanceOf[Seq[Double]].toArray.take(3)))

    new RankingMetrics(a.rdd)
  }

  /**
   *
   * @param indexedReviewDF indexed DataFrame from raw
   * @param userSubsetRecs recommendForUserSubset from ALS model
   * @param spark spark session
   * @return
   */
  def addUserAndBusinessNames(indexedReviewDF: DataFrame, userSubsetRecs: DataFrame, spark: SparkSession): DataFrame = {
    // retrieve userId, userName and select user_no and name
    val originUserDF = spark.read.format("json").load("yelp-dataset/yelp_academic_dataset_user.json")
    val usersDF = originUserDF
      .join(indexedReviewDF, originUserDF.col("user_id") === indexedReviewDF.col("user_id"))
      .select(col("user_no"), col("name")).distinct()

    // retrieve businessId, name and select business_no and name
    val originBusinessDF = spark.read.format("json").load("yelp-dataset/yelp_academic_dataset_business.json")
    val businessDF = originBusinessDF
      .join(indexedReviewDF, originBusinessDF.col("business_id") === indexedReviewDF.col("business_id"))
      .select(col("business_no"), col("name")).distinct()

    // explode recommendations
    val explodedDF = userSubsetRecs
      .withColumn("reco", explode(col("recommendations")))
      .select(col("user_no"), col("reco"))

    // join user name and business name on ALS result
    explodedDF
      .join(usersDF, explodedDF.col("user_no") === usersDF.col("user_no"), "left_outer")
      .join(businessDF, explodedDF.col("reco.business_no") === businessDF.col("business_no"), "left_outer")
      .select(explodedDF.col("user_no"), usersDF.col("name"), explodedDF.col("reco"), businessDF.col("name"))
      .orderBy(explodedDF.col("user_no"), explodedDF.col("reco.rating"))
  }
}
