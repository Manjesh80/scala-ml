package com.manjesh.sparkml.processing

import com.manjesh.sparkml.util.Util
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * Created by cloudera on 5/28/17.
  */
object SparkTemplate {

  def main(args: Array[String]): Unit = {

    Util.init("app-name")


    Util.spark.stop()
  }
}
