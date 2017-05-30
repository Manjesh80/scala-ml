package com.manjesh.sparkml.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by cloudera on 5/29/17.
  */
object SVMPipeline {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[2]", "Classification")
    Logger.getLogger("org").setLevel(Level.ERROR)
    val records = sc
      .textFile("/home/cloudera/workspace/scala-ml/src/main/scala/com/manjesh/sparkml/dataset/train_noheader.tsv")
      .map(line => line.split("\t"))

    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }

    // params for SVM
    val numIterations = 10

    // Run training algorithm to build the model
    val svmModel = SVMWithSGD.train(data, numIterations)

    // Clear the default threshold.
    svmModel.clearThreshold()

    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    // calculate accuracy
    val svmAccuracy = svmTotalCorrect / data.count()
    println(svmAccuracy)
  }

}
