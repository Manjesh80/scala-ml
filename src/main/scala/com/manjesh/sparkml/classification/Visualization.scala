package com.manjesh.sparkml.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
  * Created by cloudera on 5/29/17.
  */
object Visualization {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[2]", "Classification")
    Logger.getLogger("org").setLevel(Level.ERROR)
    var records = sc.textFile("/home/cloudera/workspace/scala-ml/src/main/scala/com/manjesh/sparkml/dataset/train_noheader.tsv")
      .map(line => line.split("\t"))

    val data_persistent = records.map(r => {
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      //println(trimmed.slice(4, r.size - 1).mkString(" ##"))
      val features = trimmed.slice(4, r.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)

      val len = features.size.toInt
      val len_2 = math.floor(len / 2).toInt

      val x = features.slice(0, len_2)
      val y = features.slice(len_2 - 1, len)

      var i = 0
      var sum_x = 0.0
      var sum_y = 0.0

      while (i < x.length) {
        sum_x += x(i)
        i += 1
      }

      i = 0
      while (i < y.length) {
        sum_y += y(i)
        i += 1
      }

      if (sum_y != 0.0) {
        if (sum_x != 0.0) {
          math.log(sum_x) + "," + math.log(sum_y)
        } else {
          sum_x + "," + math.log(sum_y)
        }
      } else {
        if (sum_x != 0.0) {
          math.log(sum_x) + "," + 0.0
        } else {
          sum_x + "," + 0.0
        }
      }
    })

    val dataone = data_persistent.first()
    println(data_persistent.saveAsTextFile("/results/raw-input-log"))

    sc.stop()
  }
}
