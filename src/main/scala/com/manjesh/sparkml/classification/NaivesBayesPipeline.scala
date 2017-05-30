package com.manjesh.sparkml.classification

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
  * Created by cloudera on 5/30/17.
  */
object NaivesBayesPipeline {

  def naiveBayesPipeline(features: VectorAssembler, dataFrame: DataFrame) = {

    val Array(training, test) = dataFrame.randomSplit(Array(0.9, 0.1), seed = 12345)

    //create Pipeline
    val stages = mutable.ArrayBuffer[PipelineStage]()

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    val naiveBayes = new NaiveBayes()

    stages += features
    stages += naiveBayes
    val pipeline = new Pipeline().setStages(stages.toArray)

    val startTime = System.nanoTime()
    val model = pipeline.fit(dataFrame)
    val elapsedTime = System.nanoTime() - startTime / 1e9
    println(s"Training time: $elapsedTime seconds")

    val holdout = model.transform(dataFrame).select("prediction","label")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val mAccuracy = evaluator.evaluate(holdout)
    println("Test set accuracy = " + mAccuracy)
  }

}




















