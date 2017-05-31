package com.manjesh.sparkml.classification

import com.manjesh.sparkml.classification.ClassificationComparisionEngine.reports
import com.manjesh.sparkml.classification.ClassificationUtils.SparkMaster
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
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
import org.apache.spark.mllib.tree.model.DecisionTreeModel

import scala.collection.mutable.ListBuffer
//import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, LinkedList}

object ClassificationComparisionEngine {

  val reports: ArrayBuffer[String] = new ArrayBuffer[String]()
  val finalReports: ArrayBuffer[String] = new ArrayBuffer[String]()
  val numIterations = 50
  val maxTreeDepth = 5
  var sparkSession: SparkSession = null

  def main(args: Array[String]): Unit = {

    //runLogisticRegressionWithSGD()
    runDecisionTree
    reports.foreach(println(_))
  }

  def runLogisticRegressionWithSGD() = {

    val rawKaggleRecords = ClassificationComparisionUtils.getKaggleRecords("Logistic-Regression")
    val allKaggleRecords = ClassificationComparisionUtils.getDataRecordsForNonNaiveBayes(rawKaggleRecords)
    allKaggleRecords.cache()

    val lrModel = LogisticRegressionWithSGD.train(allKaggleRecords, numIterations)
    reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
      " Basic Trained Model", lrModel, allKaggleRecords)

    val allKaggleScaledRecords = ClassificationComparisionUtils.scaleFeatures(allKaggleRecords)
    val lrScaledModel = LogisticRegressionWithSGD.train(allKaggleScaledRecords, numIterations)
    reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
      " Features Scaled Model ", lrScaledModel, allKaggleScaledRecords)

    val scaledDataCatagories = ClassificationComparisionUtils.categoryTunedFeatures(rawKaggleRecords)
    scaledDataCatagories.cache()
    val lrModelScaledCategories = LogisticRegressionWithSGD.train(scaledDataCatagories, numIterations)
    reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
      " Category Scaled Model ", lrModelScaledCategories, scaledDataCatagories)

    Seq(1, 5, 10, 50, 100).map { param =>
      val model = ClassificationComparisionUtils.trainLogisticRegressionWithParams(
        scaledDataCatagories, 0.0, param, new SimpleUpdater, 1.0)
      reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
        s" Category Scaled Model - $param iterations ", model, scaledDataCatagories)
    }

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = ClassificationComparisionUtils.trainLogisticRegressionWithParams(
        scaledDataCatagories, 0.0, numIterations, new SimpleUpdater, param)
      reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
        s" Category Scaled Model - $param Step-Size ", model, scaledDataCatagories)
    }

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = ClassificationComparisionUtils.trainLogisticRegressionWithParams(
        scaledDataCatagories, param, numIterations, new SimpleUpdater, 10.0)
      reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
        s" Category Scaled Model - $param L2 regularization parameter ", model, scaledDataCatagories)
    }

    sparkSession.stop()
  }

  def runDecisionTree() = {

    val rawKaggleRecords = ClassificationComparisionUtils.getKaggleRecords("Logistic-Regression")
    val allKaggleRecords = ClassificationComparisionUtils.getDataRecordsForNonNaiveBayes(rawKaggleRecords)
    allKaggleRecords.cache()

    val lrModel: DecisionTreeModel = DecisionTree.train(allKaggleRecords, Algo.Classification, Entropy, maxTreeDepth)
    reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
      " Basic Trained Model", lrModel, allKaggleRecords)

    /*val allKaggleScaledRecords = ClassificationComparisionUtils.scaleFeatures(allKaggleRecords)
    val lrScaledModel = LogisticRegressionWithSGD.train(allKaggleScaledRecords, numIterations)
    reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
      " Features Scaled Model ", lrScaledModel, allKaggleScaledRecords)

    val scaledDataCatagories = ClassificationComparisionUtils.categoryTunedFeatures(rawKaggleRecords)
    scaledDataCatagories.cache()
    val lrModelScaledCategories = LogisticRegressionWithSGD.train(scaledDataCatagories, numIterations)
    reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
      " Category Scaled Model ", lrModelScaledCategories, scaledDataCatagories)

    Seq(1, 5, 10, 50, 100).map { param =>
      val model = ClassificationComparisionUtils.trainLogisticRegressionWithParams(
        scaledDataCatagories, 0.0, param, new SimpleUpdater, 1.0)
      reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
        s" Category Scaled Model - $param iterations ", model, scaledDataCatagories)
    }

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = ClassificationComparisionUtils.trainLogisticRegressionWithParams(
        scaledDataCatagories, 0.0, numIterations, new SimpleUpdater, param)
      reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
        s" Category Scaled Model - $param Step-Size ", model, scaledDataCatagories)
    }

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = ClassificationComparisionUtils.trainLogisticRegressionWithParams(
        scaledDataCatagories, param, numIterations, new SimpleUpdater, 10.0)
      reports += ClassificationComparisionUtils.runPredictionAndGatherMetrics(
        s" Category Scaled Model - $param L2 regularization parameter ", model, scaledDataCatagories)
    }*/

    sparkSession.stop()
  }

}


object ClassificationComparisionUtils {

  def trainLogisticRegressionWithParams(input: RDD[LabeledPoint], regParam: Double,
                                        numIterations: Int, updater: Updater, stepSize: Double): ClassificationModel = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    return lr.run(input)
  }

  // helper function to create AUC metric

  def runSinglePrediction(model: ClassificationModel, allKaggleRecords: RDD[LabeledPoint]): Seq[String] = {
    val dataPoint = allKaggleRecords.first();
    val prediction = model.predict(dataPoint.features)
    val result = new ListBuffer[String]
    result += "Single Predicted Value in LR ==> " + prediction
    result += "Single Actual Value in LR ==> " + dataPoint.label
    return result
  }

  def runAccuracyAndPrediction(model: ClassificationModel, allKaggleRecords: RDD[LabeledPoint]): Seq[String] = {
    val result = new ListBuffer[String]
    val totalTP = allKaggleRecords.map {
      record => if (model.predict(record.features) == record.label) 1 else 0
    }.sum()

    val lrAccuracy = totalTP / allKaggleRecords.count()

    result += "LR :: Bulk prediction total correct ==> " + totalTP.toString
    result += "LR :: Total records ==> " + allKaggleRecords.count()
    result += "LR :: Accuracy ==> " + lrAccuracy.toString

    return result
  }

  def runPredictionAndGatherMetrics(modelType: String, model: ClassificationModel, allKaggleRecords: RDD[LabeledPoint]): String = {

    val result = new ListBuffer[String]

    val scoreAndLabels = allKaggleRecords.map {
      record => (model.predict(record.features), record.label)
    }

    val totalTP = scoreAndLabels.map { r => if (r._1 == r._2) 1 else 0 }.sum()
    val lrAccuracy = totalTP / allKaggleRecords.count()
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)

    return f"Model Type ==> ${modelType} " +
      f" **** ${model.getClass.getSimpleName} " +
      " **** Total correct ==> " + totalTP.toString +
      " **** Total records ==> " + allKaggleRecords.count() +
      f" **** Accuracy: ${lrAccuracy * 100}%2.4f%% " +
      f" **** Area under PR: ${metrics.areaUnderPR * 100.0}%2.4f%% " +
      f" **** Area under ROC: ${metrics.areaUnderROC * 100.0}%2.4f%%"
  }

  def runPredictionAndGatherMetrics(modelType: String, model: DecisionTreeModel, allKaggleRecords: RDD[LabeledPoint]): String = {

    val result = new ListBuffer[String]

    val scoreAndLabels = allKaggleRecords.map {
      record => {
        val score = model.predict(record.features)
        (if (score > 0.5) 1.0 else 0.0, record.label)
      }
    }

    val totalTP = scoreAndLabels.map { r => if (r._1 == r._2) 1 else 0 }.sum()
    val lrAccuracy = totalTP / allKaggleRecords.count()
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)

    return f"Model Type ==> ${modelType} " +
      f" **** ${model.getClass.getSimpleName} " +
      " **** Total correct ==> " + totalTP.toString +
      " **** Total records ==> " + allKaggleRecords.count() +
      f" **** Accuracy: ${lrAccuracy * 100}%2.4f%% " +
      f" **** Area under PR: ${metrics.areaUnderPR * 100.0}%2.4f%% " +
      f" **** Area under ROC: ${metrics.areaUnderROC * 100.0}%2.4f%%"
  }

  def categoryTunedFeatures(rawKaggleRecords: RDD[Array[String]]): RDD[LabeledPoint] = {

    val categories = rawKaggleRecords.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
    /*reports += (" All categories in Kaggle " + categories)
    reports += (" Total number of Categories " + numCategories)*/

    val dataCategories = ClassificationComparisionUtils.getDataCategory(rawKaggleRecords)
    val scalerCategories = new StandardScaler(withMean = true, withStd = true).
      fit(dataCategories.map(lp => lp.features))

    return dataCategories.map(lp => LabeledPoint(lp.label, scalerCategories.transform(lp.features)))

  }

  def scaleFeatures(allKaggleRecords: RDD[LabeledPoint]): RDD[LabeledPoint] = {

    val allFeatureVector = allKaggleRecords.map(_.features)
    val featuresRowMatrix = new RowMatrix(allFeatureVector)
    val featuresRowMatrixColumnarSummary = featuresRowMatrix.computeColumnSummaryStatistics()

    /*reports += " Mean of feature vector" + featuresRowMatrixColumnarSummary.mean
    reports += " Min of feature vector" + featuresRowMatrixColumnarSummary.min
    reports += " Max of feature vector" + featuresRowMatrixColumnarSummary.max
    reports += " Variance of feature vector" + featuresRowMatrixColumnarSummary.variance
    reports += " Non-Zeros of feature vector" + featuresRowMatrixColumnarSummary.numNonzeros*/

    val allFeaturesScaler = new StandardScaler(withMean = true, withStd = true).fit(allFeatureVector)
    val allKaggleScaledRecords = allKaggleRecords.map(
      lp => LabeledPoint(lp.label, allFeaturesScaler.transform(lp.features)))
    return allKaggleScaledRecords
  }

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

  def getKaggleRecords(appName: String): RDD[Array[String]] = {
    val appName = "Logistic-Regression"
    val sparkSession = ClassificationComparisionUtils.getSparkSession(appName)
    val sparkContext = ClassificationComparisionUtils.getSparkContext(sparkSession)
    val rawKaggleRecords = ClassificationComparisionUtils.getKaggleRecords(sparkContext)
    ClassificationComparisionEngine.sparkSession = sparkSession
    return rawKaggleRecords
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


/*

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

       //endregion*/