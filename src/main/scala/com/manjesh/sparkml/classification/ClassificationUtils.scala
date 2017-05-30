package com.manjesh.sparkml.classification

import org.apache.spark.SparkConf

/**
  * Created by cloudera on 5/30/17.
  */

object ClassificationUtils {

  val SparkMaster = "local[2]"

  def createSparkConf(appName: String): SparkConf = {
    new SparkConf().setAppName(appName).setMaster(SparkMaster)
  }

}
