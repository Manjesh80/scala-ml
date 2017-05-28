package com.manjesh.sparkml.processing

import com.manjesh.sparkml.util.ProcessingUtils
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.log4j.{Level, Logger}

import scala.collection.immutable.ListMap
import scalax.chart.module.ChartFactories

/**
  * Created by cloudera on 5/27/17.
  */
object LoadUserData {


  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    val spConfig = (new SparkConf).setMaster("local").setAppName("SparkApp")
    val sparkSession = SparkSession
      .builder()
      .appName("SparkUserData").config(spConfig)
      .getOrCreate()

    val user_df = sparkSession.read.format("com.databricks.spark.csv")
      .option("delimiter", "|").schema(ProcessingUtils.getUserSchema)
      .load("/home/cloudera/workspace/scala-ml/data/u.user")

    val first = user_df.first()
    println("First Record : " + first)
    println("User Count ==> " + user_df.count())

    println("Group by Gender --- \r\n")
    println(user_df.groupBy("gender").count().collect());

    val genderRows: Array[Row] = user_df.groupBy("gender").count().collect();
    genderRows.map(r => {
      println(r.get(0) + " == > " + r.get(1))
      println("")
    })
    println("Group by Gender --- \r\n")
    println(user_df.groupBy("gender").count().collect().length);
    println("Group by Gender --- \r\n")

    // Print Histogram

    val ages_array = user_df.select("age").collect()

    val min = 0
    val max = 80
    val bins = 16
    val step = (80 / bins).toInt
    var mx = Map(0 -> 0)

    for (i <- step until (max + step) by step) {
      mx += (i -> 0)
    }

    for (x <- 0 until ages_array.length) {
      val age = Integer.parseInt(ages_array(x)(0).toString)
      for (j <- 0 until (max + step) by step) {
        if (age > j && age <= j + step) {
          mx = mx + (j -> (mx(j) + 1))
        }
      }
    }

    val mx_sorted = ListMap(mx.toSeq.sortBy(_._1): _*)

    val ds = new org.jfree.data.category.DefaultCategoryDataset
    mx_sorted.foreach {
      case (k, v) => {
        ds.addValue(v, "UserAges", k)
      }
    }
    val chart = ChartFactories.BarChart(ds)
    chart.show()

    sparkSession.stop()
  }


}
