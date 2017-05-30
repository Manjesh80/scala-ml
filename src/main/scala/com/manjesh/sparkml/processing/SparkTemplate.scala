package com.manjesh.sparkml.processing

import com.manjesh.sparkml.util.Util

/**
  * Created by cloudera on 5/28/17.
  */
object SparkTemplate {

  def main(args: Array[String]): Unit = {
    Util.init("load-movie-data")

    Util.spark.stop()
  }


}
