package com.manjesh.sparkml.recommendation


import com.manjesh.sparkml.model.Rating
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.ml.evaluation._
import org.apache.spark.mllib.evaluation._

/**
  * Created by cloudera on 5/28/17.
  */
object MovieRecommendation {

  var sparkSession: SparkSession = null

  def main(args: Array[String]): Unit = {
    getFeatures
    createALSModel
  }

  // Alternating Least Squares
  def createALSModel() {
    val ratings = getFeatures()

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      //.setImplicitPrefs(true)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(training)

    val predictions = model.transform(test)
    println(predictions.printSchema())

    // Not working in 2.0
    /*val predictedRating = model.predict(789, 123)
    val userId = 789
    val K = 10
    val topKRecs = model.recommendProducts(userId, K)
    println(topKRecs.mkString("n"))*/

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    println(s"Root-mean-square error = $rmse")



    /*println(model)
    println(model.userFactors.schema)
    println(model.itemFactors.schema)
    model.userFactors.show
    model.itemFactors.show*/

  }


  def getFeatures(): Dataset[Rating] = {

    val spark = SparkSession.builder.master("local[2]")
      .appName("FeatureExtraction").getOrCreate()
    Logger.getLogger("org").setLevel(Level.WARN)

    sparkSession = spark
    import spark.implicits._

    val ratings = spark.read.textFile("/home/cloudera/workspace/scala-ml/data/u.data")
      .map(parseRating)

    println(ratings.first())

    return ratings
  }


  def parseRating(str: String): Rating = {
    val fields = str.split("\t")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
  }

}
