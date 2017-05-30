package com.manjesh.sparkml.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors

/**
  * Created by cloudera on 5/29/17.
  */
object LinearSVM {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[2]", "Classification")
    Logger.getLogger("org").setLevel(Level.ERROR)
    var records = sc.textFile("/home/cloudera/workspace/scala-ml/src/main/scala/com/manjesh/sparkml/dataset/train_noheader.tsv")
      .map(line => line.split("\t"))

    val data_persistent = records.map(r => {
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt

      //Bayes model requires non-negative features, and will
      // throw an error if it encounters negative values.

      val features = trimmed.slice(4, r.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))


    })

    sc.stop()
  }
}
