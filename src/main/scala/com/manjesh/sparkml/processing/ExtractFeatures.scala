package com.manjesh.sparkml.processing

import breeze.linalg.DenseVector
import com.manjesh.sparkml.util.Util

/**
  * Created by cloudera on 5/28/17.
  */
object ExtractFeatures {

  def main(args: Array[String]): Unit = {

    Util.init("extract-features")

    val rating_df = Util.getRatingDF();
    val user_df = Util.getUserDF();
    val movie_df = Util.getMovieDF();

    val ratings_grouped = rating_df.groupBy("rating")
    ratings_grouped.count().show()

    val occupation_df = user_df.select("occupation").distinct().sort("occupation").collect()

    var all_occupation_dict: Map[String, Int] = Map()

    for (i <- 0 to occupation_df.length - 1) {
      all_occupation_dict += occupation_df(i)(0).toString -> i
    }

    println(all_occupation_dict.toSeq.sortBy(_._2))

    val occupationSize = all_occupation_dict.size;
    val occupation_feature = DenseVector.zeros[Double](occupationSize)
    occupation_feature(all_occupation_dict("doctor")) = 1

    println("Occupation feature vector: %s" + occupation_feature)
    println("Length of binary vector: " + occupation_feature.activeSize)

    Util.spark.stop()
  }
}
