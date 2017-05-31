package com.manjesh.sparkml.classification

import com.manjesh.sparkml.classification.ClassificationUtils.SparkMaster
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
//import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, LinkedList}

object ClassificationComparisionEngine {

  val reports: ArrayBuffer[String] = new ArrayBuffer[String]()
  val finalReports: ArrayBuffer[String] = new ArrayBuffer[String]()

  def main(args: Array[String]): Unit = {

    runLogisticRegression()
    reports.foreach(println(_))

    println("Final Report \n\n\n")
    finalReports.foreach(println(_))

    /*val appName = "Logistic-Regression"
    val sparkSession = ClassificationComparisionUtils.getSparkSession(appName)
    val kaggleRecords = ClassificationComparisionUtils.getKaggleRecords(sparkSession.sparkContext)
    val nonNaviesBayesRecords = ClassificationComparisionUtils.getDataRecordsForNonNaiveBayes(kaggleRecords)
    val naviesBayesRecords = ClassificationComparisionUtils.getDataRecordsForNaiveBayes(kaggleRecords)*/


  }

  def runLogisticRegression() = {

    //region Initial Model Load

    val appName = "Logistic-Regression"

    val sparkSession = ClassificationComparisionUtils.getSparkSession(appName)
    val sparkContext = ClassificationComparisionUtils.getSparkContext(sparkSession)
    val rawKaggleRecords = ClassificationComparisionUtils.getKaggleRecords(sparkContext)

    val allKaggleRecords = ClassificationComparisionUtils.getDataRecordsForNonNaiveBayes(rawKaggleRecords)
    allKaggleRecords.cache()
    val numIterations = 50
    val lrModel = LogisticRegressionWithSGD.train(allKaggleRecords, numIterations)

    //endregion

    //region Single Data Point prediction

    val dataPoint = allKaggleRecords.first();
    val prediction = lrModel.predict(dataPoint.features)
    reports += "Single Predicted Value in LR ==> " + prediction
    reports += "Single Actual Value in LR ==> " + dataPoint.label

    //endregion

    //region Bulk Prediction
    //val bulkPredictions = lrModel.predict(allKaggleRecords.map(_.features))
    //endregion

    //region Accuracy and prediction error

    val totalTP = allKaggleRecords.map {
      record => if (lrModel.predict(record.features) == record.label) 1 else 0
    }.sum()


    reports += "LR :: Bulk prediction total correct ==> " + totalTP.toString
    reports += "LR :: Total records ==> " + allKaggleRecords.count()

    val lrAccuracy = totalTP / allKaggleRecords.count()
    reports += "LR :: Accuracy ==> " + lrAccuracy.toString

    //endregion

    //region Calculate PR and ROC

    val lrMetrics = Seq(lrModel).map { model =>

      val scoreAndLabels = allKaggleRecords.map {
        point => (model.predict(point.features), point.label)
      }

      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val basicReports: String = " Regular Training ==> " + (f"${lrMetrics(0)._1} " +
      f"**** Accuracy: ${lrAccuracy * 100}%2.4f%% " +
      f"**** Area under PR: ${lrMetrics(0)._2 * 100.0}%2.4f%% " +
      f"**** Area under ROC: ${lrMetrics(0)._3 * 100.0}%2.4f%%")

    reports += basicReports
    finalReports += basicReports

    /*lrMetrics.foreach {
      case (model, pr, roc) =>
        reports += ((f"$model, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%"))
    }*/

    //endregion

    //region Understand features Data with RowMatrix, and scale features

    val allFeatureVector = allKaggleRecords.map(_.features)
    val featuresRowMatrix = new RowMatrix(allFeatureVector)
    val featuresRowMatrixColumnarSummary = featuresRowMatrix.computeColumnSummaryStatistics()

    reports += " Mean of feature vector" + featuresRowMatrixColumnarSummary.mean
    reports += " Min of feature vector" + featuresRowMatrixColumnarSummary.min
    reports += " Max of feature vector" + featuresRowMatrixColumnarSummary.max
    reports += " Variance of feature vector" + featuresRowMatrixColumnarSummary.variance
    reports += " Non-Zeros of feature vector" + featuresRowMatrixColumnarSummary.numNonzeros

    val allFeaturesScaler = new StandardScaler(withMean = true, withStd = true).fit(allFeatureVector)
    val allKaggleScaledRecords = allKaggleRecords.map(
      lp => LabeledPoint(lp.label, allFeaturesScaler.transform(lp.features)))

    //endregion

    //region Train Scaled Model

    reports += "LR :: Scaling the features and training "

    val lrScaledModel = LogisticRegressionWithSGD.train(allKaggleScaledRecords, numIterations)
    val lrTotalCorrectScaled = allKaggleScaledRecords.map { point =>
      if (lrScaledModel.predict(point.features) == point.label) 1 else
        0
    }.sum

    val lrAccuracyScaled = lrTotalCorrectScaled / allKaggleScaledRecords.count()

    reports += "LR :: Accuracy after scaling ==> " + lrAccuracy.toString

    val lrScaledPredictionsVsTrue = allKaggleScaledRecords.map {
      point => (lrScaledModel.predict(point.features), point.label)
    }

    val lrMetricsScaled = new BinaryClassificationMetrics(lrScaledPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC
    val scaledReports = " After scaling features ==> " + (f"${lrScaledModel.getClass.getSimpleName} " +
      f"**** Accuracy: ${lrAccuracyScaled * 100}%2.4f%% " +
      f"**** Area under PR: ${lrPr * 100.0}%2.4f%% " +
      f"**** Area under ROC: ${lrRoc * 100.0}%2.4f%%")

    reports += scaledReports
    finalReports += scaledReports

    //endregion

    //region Tune on Category

    val categories = rawKaggleRecords.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
    reports += (" All categories in Kaggle " + categories)
    reports += (" Total number of Categories " + numCategories)

    val dataCategories = ClassificationComparisionUtils.getDataCategory(rawKaggleRecords)
    val scalerCategories = new StandardScaler(withMean = true, withStd = true).
      fit(dataCategories.map(lp => lp.features))

    val scaledDataCatagories =
      dataCategories.map(lp => LabeledPoint(lp.label, scalerCategories.transform(lp.features)))

    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCatagories, numIterations)

    val lrTotalCorrectScaledCats = scaledDataCatagories.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else
        0
    }.sum

    val lrAccuracyScaledCats = lrTotalCorrectScaledCats / allKaggleScaledRecords.count()
    val lrPredictionsVsTrueCats = scaledDataCatagories.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR
    val lrRocCats = lrMetricsScaledCats.areaUnderROC
    val CategorySacledReports: String = " After scaling category ==> " +
      (f"${lrModelScaledCats.getClass.getSimpleName} " +
        f"**** Accuracy: ${lrAccuracyScaledCats * 100}%2.4f%% " +
        f"**** Area under PR: ${lrPrCats * 100.0}%2.4f%% " +
        f"**** Area under ROC: ${lrRocCats * 100.0}%2.4f%%")

    reports += CategorySacledReports
    finalReports += CategorySacledReports

    //endregion

    scaledDataCatagories.cache()

    //region Train LR model with various iterations

    val iterationResults = Seq(1, 5, 10, 50, 100).map { param =>
      val model = ClassificationComparisionUtils.trainWithParams(
        scaledDataCatagories, 0.0, param, new SimpleUpdater, 1.0)
      ClassificationComparisionUtils.createMetrics(s"$param iterations", scaledDataCatagories, model)
    }

    iterationResults.foreach {
      case (param, auc, pr) => {
        val categoryScaledReports: String = " After scaling category ==> " +
          (f"${lrModelScaledCats.getClass.getSimpleName} " +
            //f"**** Accuracy: ${lrAccuracyScaledCats * 100}%2.4f%% " +
            f"**** Param: $param " +
            f"**** Accuracy: N/A " +
            f"**** Area under PR: ${pr * 100.0}%2.4f%% " +
            f"**** Area under ROC: ${auc * 100.0}%2.4f%% ")

        finalReports += categoryScaledReports
      }
    }

    //endregion

    //region Train LR model with various step size

    val stepSizeResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = ClassificationComparisionUtils.trainWithParams(
        scaledDataCatagories, 0.0, numIterations, new SimpleUpdater, param)
      ClassificationComparisionUtils.createMetrics(s"$param step-size", scaledDataCatagories, model)
    }

    stepSizeResults.foreach {
      case (param, auc, pr) => {
        val categoryScaledReports: String = " After scaling category ==> " +
          (f"${lrModelScaledCats.getClass.getSimpleName} " +
            //f"**** Accuracy: ${lrAccuracyScaledCats * 100}%2.4f%% " +
            f"**** Param: $param " +
            f"**** Accuracy: N/A " +
            f"**** Area under PR: ${pr * 100.0}%2.4f%% " +
            f"**** Area under ROC: ${auc * 100.0}%2.4f%% ")

        finalReports += categoryScaledReports
      }
    }

    //endregion

    //region Train LR with Regularization

    val regularizationResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = ClassificationComparisionUtils.trainWithParams(
        scaledDataCatagories, param, numIterations, new SimpleUpdater, 10.0)
      ClassificationComparisionUtils.createMetrics(s"$param L2 regularization parameter", scaledDataCatagories, model)
    }

    regularizationResults.foreach {
      case (param, auc, pr) => {
        val categoryScaledReports: String = " After scaling category ==> " +
          (f"${lrModelScaledCats.getClass.getSimpleName} " +
            f"**** Param: $param " +
            f"**** Accuracy: N/A " +
            f"**** Area under PR: ${pr * 100.0}%2.4f%% " +
            f"**** Area under ROC: ${auc * 100.0}%2.4f%% ")

        finalReports += categoryScaledReports
      }
    }

    //endregion

    //region Illustrate Cross Validation

    val trainTestSplit = scaledDataCatagories.randomSplit(Array(0.60, 0.4), 123)
    val train = trainTestSplit(0)
    val test = trainTestSplit(1)

    // now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
    // in addition, we will evaluate the differing performance of regularization on training and test datasets
    val regularizationTestResults = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      val model = ClassificationComparisionUtils.trainWithParams(
                train, param, numIterations, new SquaredL2Updater, 1.0)
      ClassificationComparisionUtils.createMetrics(s"$param L2 regularization parameter", test, model)
    }

    regularizationTestResults.foreach {
      case (param, auc, pr) => {
        val categoryScaledReports: String = " After scaling category with 60/40 - Test ==> " +
          (f"${lrModelScaledCats.getClass.getSimpleName} " +
            f"**** Param: $param " +
            f"**** Accuracy: N/A " +
            f"**** Area under PR: ${pr * 100.0}%2.4f%% " +
            f"**** Area under ROC: ${auc * 100.0}%2.4f%% ")

        finalReports += categoryScaledReports
      }
    }

    val regularizationTrainResults = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      val model = ClassificationComparisionUtils.trainWithParams(
        train, param, numIterations, new SquaredL2Updater, 1.0)
      ClassificationComparisionUtils.createMetrics(s"$param L2 regularization parameter", train, model)
    }

    regularizationTrainResults.foreach {
      case (param, auc, pr) => {
        val categoryScaledReports: String = " After scaling category with 60/40 - Train ==> " +
          (f"${lrModelScaledCats.getClass.getSimpleName} " +
            f"**** Param: $param " +
            f"**** Accuracy: N/A " +
            f"**** Area under PR: ${pr * 100.0}%2.4f%% " +
            f"**** Area under ROC: ${auc * 100.0}%2.4f%% ")

        finalReports += categoryScaledReports
      }
    }

    //endregion

    sparkSession.stop()
  }
}


object ClassificationComparisionUtils {

  def trainWithParams(input: RDD[LabeledPoint], regParam: Double,
                      numIterations: Int, updater: Updater, stepSize: Double) = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
  }

  // helper function to create AUC metric

  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC, metrics.areaUnderPR())
  }

  def getSparkConf(appName: String): SparkConf = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.OFF)
    new SparkConf().setAppName(appName).setMaster(SparkMaster)
  }

  def getSparkSession(appName: String): SparkSession = {
    return SparkSession.builder().config(getSparkConf(appName)).getOrCreate()
  }

  def getSparkSession(sparkConf: SparkConf): SparkSession = {
    return SparkSession.builder().config(sparkConf).getOrCreate()
  }

  def getSparkContext(sparkSession: SparkSession): SparkContext = {
    val sc = sparkSession.sparkContext
    sc.setLogLevel("WARN")
    return sc
  }

  def getSparkContext(appName: String): SparkContext = {
    val sc = SparkSession.builder().config(getSparkConf(appName)).getOrCreate().sparkContext
    sc.setLogLevel("WARN")
    return sc
  }

  def getKaggleRecords(sparkContext: SparkContext): RDD[Array[String]] = {
    sparkContext
      .textFile("/home/cloudera/workspace/scala-ml/src/main/scala/com/manjesh/sparkml/dataset/train_noheader.tsv")
      .map(line => line.split("\t"))
  }

  def getDataRecordsForNonNaiveBayes(kaggleRecords: RDD[Array[String]]): RDD[LabeledPoint] = {
    kaggleRecords.map {
      kaggleRecord => {
        val trimmed = kaggleRecord.map(_.replaceAll("\"", ""))
        val label = trimmed(kaggleRecord.size - 1).toInt
        val features = trimmed.slice(4, kaggleRecord.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }
    }
  }

  def getDataRecordsForNaiveBayes(kaggleRecords: RDD[Array[String]]): RDD[LabeledPoint] = {
    kaggleRecords.map {
      kaggleRecord => {
        val trimmed = kaggleRecord.map(_.replaceAll("\"", ""))
        val label = trimmed(kaggleRecord.size - 1).toInt
        val features = trimmed
          .slice(4, kaggleRecord.size - 1)
          .map(d => if (d == "?") 0.0 else d.toDouble)
          .map(d => if (d < 0) 0.0 else d)
        LabeledPoint(label, Vectors.dense(features))
      }
    }
  }

  def getDataCategory(rawKaggleRecords: RDD[Array[String]]): RDD[LabeledPoint] = {

    val categories = rawKaggleRecords.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size

    val dataCategories = rawKaggleRecords.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if
      (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }

    return dataCategories
  }
}
