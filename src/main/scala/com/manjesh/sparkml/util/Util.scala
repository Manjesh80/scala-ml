package com.manjesh.sparkml.util

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

/**
  * Created by cloudera on 5/27/17.
  */
object Util {

  var sparkConf: SparkConf = null
  var sparkSession: SparkSession = null

  //region Init

  def init(appName: String): SparkSession = {
    Logger.getLogger("org").setLevel(Level.WARN)
    sparkConf = (new SparkConf()).setMaster("local[2]").setAppName(appName)
    sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    return sparkSession
  }

  def spark = sparkSession

  //endregion

  //region Schemas

  def getUserSchema: StructType = {

    return StructType(
      Array(
        StructField("no", IntegerType, true),
        StructField("age", StringType, true),
        StructField("gender", StringType, true),
        StructField("occupation", StringType, true),
        StructField("zipCode", StringType, true)
      )
    )
  }

  def getMovieSchema: StructType = StructType(
    Array(
      StructField("id", StringType, true),
      StructField("name", StringType, true),
      StructField("date", StringType, true),
      StructField("unknown", StringType, true),
      StructField("url", StringType, true)
    )
  )

  def getRatingSchema: StructType = StructType(
    Array(
      StructField("user_id", IntegerType, true),
      StructField("movie_id", IntegerType, true),
      StructField("rating", IntegerType, true),
      StructField("timestamp", IntegerType, true)
    )
  )

  //endregion

  //region DataFramres

  def getUserDF(): DataFrame = {
    return sparkSession.read
      .format("com.databricks.spark.csv")
      .option("delimiter", "|")
      .schema(getUserSchema)
      .load("/home/cloudera/workspace/scala-ml/data/u.user")
  }

  def getMovieDF(): DataFrame = {
    return sparkSession.read
      .format("com.databricks.spark.csv")
      .option("delimiter", "|")
      .schema(getMovieSchema)
      .load("/home/cloudera/workspace/scala-ml/data/u.item")
  }

  def getRatingDF(): DataFrame = {
    return sparkSession.read
      .format("com.databricks.spark.csv")
      .option("delimiter", "\t")
      .schema(getRatingSchema)
      .load("/home/cloudera/workspace/scala-ml/data/u.data")
  }

  //endregion

  //region Misc

  def convertYear(input: String): Int = {
    try {
      return input.takeRight(4).toInt
    }
    catch {
      case e: Exception => {
        println("exception caught: " + e + " Returning 1900")
        return 1900
      }
    }
  }

  def convertRating(input: String): Int = {
    try {
      return Integer.parseInt(input)
    }
    catch {
      case e: Exception => {
        println("An execption occured " + e.getMessage)
        return 1
      }
    }
  }

  def median(input :Array[Int]) : Int = {
    val l = input.length
    val l_2 = l/2.toInt
    val x = l%2
    var y = 0
    if(x == 0) {
      y = ( input(l_2) + input(l_2 + 1))/2
    }else {
      y = (input(l_2))
    }
    return y
  }

  //endregion

}
