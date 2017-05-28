package com.manjesh.sparkml.processing

import com.manjesh.sparkml.util.Util
import breeze.linalg.{DenseVector, norm}


/**
  * Created by cloudera on 5/28/17.
  */
object LoadMovieData {

  def main(args: Array[String]): Unit = {
    Util.init("load-movie-data")

    val movie_df = Util.getMovieDF()
    movie_df.createOrReplaceTempView("movie_data")
    Util.spark.udf.register("convertYear", Util.convertYear _)

    val movieYears = getMovieYearsSorted()

    for (movie <- 0 to movieYears.length - 1)
      println(movieYears(movie))

    sanitize_data

    Util.spark.stop()
  }

  def sanitize_data = {

    val movie_years = Util.spark.sql("select convertYear(date) as year from movie_data")
    movie_years.createOrReplaceTempView("movie_years")

    val years_valid = movie_years.collect()

    val years_int = new Array[Int](years_valid.length)
    for (i <- 0 to years_valid.length - 1) {
      years_int(i) = Integer.parseInt(years_valid(i)(0).toString)
    }

    val yearsReplacedRDD = Util.spark.sparkContext.parallelize(years_int)
    val years_int_sorted = years_int.sorted

    val median_v = Util.median(years_int_sorted)

    println("Median value of Year:"+ median_v)
  }

  def getMovieYearsSorted(): scala.Array[(Int, String)] = {

    val movie_years_count = Util.spark
      .sql("select convertYear(date) as year from movie_data")
      .groupBy("year")
      .count()

    val sortedMovies = movie_years_count.rdd
      .map(row => (Integer.parseInt(row(0).toString), row(1).toString))
      .collect()
      .sortBy(_._1)

    return sortedMovies
  }
}
