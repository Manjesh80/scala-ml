package com.manjesh.sparkml.classification

import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler

/**
  * Created by cloudera on 5/30/17.
  **/

object RecommendationEngine {

  def main(args: Array[String]): Unit = {

    //val model = args(0)
    val model = "NB"

    Logger.getLogger("org").setLevel(Level.WARN)
    val sparkConf = ClassificationUtils.createSparkConf("ML-CLASSIFICATION-RECOMMENDATION-ENGINE")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    val sparkContext = sparkSession.sparkContext;
    val sqlContext = sparkSession.sqlContext;

    println(" ************* " + Calendar.getInstance().getTime().toString + " ************* ")

    val basicDF = sqlContext.read.format("com.databricks.spark.csv")
      .option("delimiter", "\t")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/home/cloudera/workspace/scala-ml/src/main/scala/com/manjesh/sparkml/dataset/train.tsv")

    basicDF.createOrReplaceTempView("StumbleUpon")
    basicDF.printSchema()

    sqlContext.sql("SELECT * FROM StumbleUpon WHERE alchemy_category = '?'")

    //region features Dataframe

    val featuresDF = basicDF.withColumn("avglinksize", basicDF("avglinksize").cast("double"))
      .withColumn("commonlinkratio_1", basicDF("commonlinkratio_1").cast("double"))
      .withColumn("commonlinkratio_2", basicDF("commonlinkratio_2").cast("double"))
      .withColumn("commonlinkratio_3", basicDF("commonlinkratio_3").cast("double"))
      .withColumn("commonlinkratio_4", basicDF("commonlinkratio_4").cast("double"))
      .withColumn("compression_ratio", basicDF("compression_ratio").cast("double"))
      .withColumn("embed_ratio", basicDF("embed_ratio").cast("double"))
      .withColumn("framebased", basicDF("framebased").cast("double"))
      .withColumn("frameTagRatio", basicDF("frameTagRatio").cast("double"))
      .withColumn("hasDomainLink", basicDF("hasDomainLink").cast("double"))
      .withColumn("html_ratio", basicDF("html_ratio").cast("double"))
      .withColumn("image_ratio", basicDF("image_ratio").cast("double"))
      .withColumn("is_news", basicDF("is_news").cast("double"))
      .withColumn("lengthyLinkDomain", basicDF("lengthyLinkDomain").cast("double"))
      .withColumn("linkwordscore", basicDF("linkwordscore").cast("double"))
      .withColumn("news_front_page", basicDF("news_front_page").cast("double"))
      .withColumn("non_markup_alphanum_characters", basicDF("non_markup_alphanum_characters").cast("double"))
      .withColumn("numberOfLinks", basicDF("numberOfLinks").cast("double"))
      .withColumn("numwords_in_url", basicDF("numwords_in_url").cast("double"))
      .withColumn("parametrizedLinkRatio", basicDF("parametrizedLinkRatio").cast("double"))
      .withColumn("spelling_errors_ratio", basicDF("spelling_errors_ratio").cast("double"))
      .withColumn("label", basicDF("label").cast("double"))

    //endregion

    //region allNumbersFeatureDF

    val replacefunc = udf { (x: Double) => if (x == "?") 0.0 else x }

    val allNumbersFeatureDF = featuresDF.withColumn("avglinksize", replacefunc(featuresDF("avglinksize")))
      .withColumn("commonlinkratio_1", replacefunc(featuresDF("commonlinkratio_1")))
      .withColumn("commonlinkratio_2", replacefunc(featuresDF("commonlinkratio_2")))
      .withColumn("commonlinkratio_3", replacefunc(featuresDF("commonlinkratio_3")))
      .withColumn("commonlinkratio_4", replacefunc(featuresDF("commonlinkratio_4")))
      .withColumn("compression_ratio", replacefunc(featuresDF("compression_ratio")))
      .withColumn("embed_ratio", replacefunc(featuresDF("embed_ratio")))
      .withColumn("framebased", replacefunc(featuresDF("framebased")))
      .withColumn("frameTagRatio", replacefunc(featuresDF("frameTagRatio")))
      .withColumn("hasDomainLink", replacefunc(featuresDF("hasDomainLink")))
      .withColumn("html_ratio", replacefunc(featuresDF("html_ratio")))
      .withColumn("image_ratio", replacefunc(featuresDF("image_ratio")))
      .withColumn("is_news", replacefunc(featuresDF("is_news")))
      .withColumn("lengthyLinkDomain", replacefunc(featuresDF("lengthyLinkDomain")))
      .withColumn("linkwordscore", replacefunc(featuresDF("linkwordscore")))
      .withColumn("news_front_page", replacefunc(featuresDF("news_front_page")))
      .withColumn("non_markup_alphanum_characters", replacefunc(featuresDF("non_markup_alphanum_characters")))
      .withColumn("numberOfLinks", replacefunc(featuresDF("numberOfLinks")))
      .withColumn("numwords_in_url", replacefunc(featuresDF("numwords_in_url")))
      .withColumn("parametrizedLinkRatio", replacefunc(featuresDF("parametrizedLinkRatio")))
      .withColumn("spelling_errors_ratio", replacefunc(featuresDF("spelling_errors_ratio")))
      .withColumn("label", replacefunc(featuresDF("label")))

    //endregion

    //region actualCleanedFeatures and allGoodCleanedFeatures

    val actualCleanedFeatures = allNumbersFeatureDF
      .drop("url")
      .drop("urlid")
      .drop("boilerplate")
      .drop("alchemy_category")
      .drop("alchemy_category_score")

    val allGoodCleanedFeatures = actualCleanedFeatures.na.fill(0.0)

    //endregion

    allGoodCleanedFeatures.createOrReplaceTempView("StumbleUpon_PreProc")

    if (model != "NB") {
      val features = getFeatures;
    }
    else {
      val features = getFeatures;
      val naivesBaseDF = prepareNaivesBaseDF(allGoodCleanedFeatures)
      runPipeline(model, features, naivesBaseDF, sparkContext)
    }

    sparkSession.stop()
  }

  def runPipeline(model: String, features: VectorAssembler,
                  dataFrame: DataFrame, sparkContext: SparkContext) = model match {

    case "NB" => {
      NaivesBayesPipeline.naiveBayesPipeline(features, dataFrame)
    }

  }

  def getFeatures(): VectorAssembler = {
    return new VectorAssembler()
      .setInputCols(Array("avglinksize", "commonlinkratio_1", "commonlinkratio_2", "commonlinkratio_3", "commonlinkratio_4", "compression_ratio"
        , "embed_ratio", "framebased", "frameTagRatio", "hasDomainLink", "html_ratio", "image_ratio"
        , "is_news", "lengthyLinkDomain", "linkwordscore", "news_front_page", "non_markup_alphanum_characters", "numberOfLinks"
        , "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"))
      .setOutputCol("features")
  }

  def prepareNaivesBaseDF(allGoodCleanedFeatures: DataFrame): DataFrame = {

    val naviesBaseReplacefunc = udf { (x: Double) => if (x < 0) 0.0 else x }

    val naivesBaseDF = allGoodCleanedFeatures.withColumn("avglinksize", naviesBaseReplacefunc(allGoodCleanedFeatures("avglinksize")))
      .withColumn("commonlinkratio_1", naviesBaseReplacefunc(allGoodCleanedFeatures("commonlinkratio_1")))
      .withColumn("commonlinkratio_2", naviesBaseReplacefunc(allGoodCleanedFeatures("commonlinkratio_2")))
      .withColumn("commonlinkratio_3", naviesBaseReplacefunc(allGoodCleanedFeatures("commonlinkratio_3")))
      .withColumn("commonlinkratio_4", naviesBaseReplacefunc(allGoodCleanedFeatures("commonlinkratio_4")))
      .withColumn("compression_ratio", naviesBaseReplacefunc(allGoodCleanedFeatures("compression_ratio")))
      .withColumn("embed_ratio", naviesBaseReplacefunc(allGoodCleanedFeatures("embed_ratio")))
      .withColumn("framebased", naviesBaseReplacefunc(allGoodCleanedFeatures("framebased")))
      .withColumn("frameTagRatio", naviesBaseReplacefunc(allGoodCleanedFeatures("frameTagRatio")))
      .withColumn("hasDomainLink", naviesBaseReplacefunc(allGoodCleanedFeatures("hasDomainLink")))
      .withColumn("html_ratio", naviesBaseReplacefunc(allGoodCleanedFeatures("html_ratio")))
      .withColumn("image_ratio", naviesBaseReplacefunc(allGoodCleanedFeatures("image_ratio")))
      .withColumn("is_news", naviesBaseReplacefunc(allGoodCleanedFeatures("is_news")))
      .withColumn("lengthyLinkDomain", naviesBaseReplacefunc(allGoodCleanedFeatures("lengthyLinkDomain")))
      .withColumn("linkwordscore", naviesBaseReplacefunc(allGoodCleanedFeatures("linkwordscore")))
      .withColumn("news_front_page", naviesBaseReplacefunc(allGoodCleanedFeatures("news_front_page")))
      .withColumn("non_markup_alphanum_characters", naviesBaseReplacefunc(allGoodCleanedFeatures("non_markup_alphanum_characters")))
      .withColumn("numberOfLinks", naviesBaseReplacefunc(allGoodCleanedFeatures("numberOfLinks")))
      .withColumn("numwords_in_url", naviesBaseReplacefunc(allGoodCleanedFeatures("numwords_in_url")))
      .withColumn("parametrizedLinkRatio", naviesBaseReplacefunc(allGoodCleanedFeatures("parametrizedLinkRatio")))
      .withColumn("spelling_errors_ratio", naviesBaseReplacefunc(allGoodCleanedFeatures("spelling_errors_ratio")))
      .withColumn("label", naviesBaseReplacefunc(allGoodCleanedFeatures("label")))

    return naivesBaseDF

  }
}
