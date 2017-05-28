package com.manjesh.sparkml.processing

import com.manjesh.sparkml.util.Util

/**
  * Created by cloudera on 5/28/17.
  */
object LoadRatingData {

  def main(args: Array[String]): Unit = {

    Util.init("load-rating-data")
    Util.spark.udf.register("convertRating", Util.convertRating _)
    val rating_df = Util.getRatingDF();
    rating_df.createOrReplaceTempView("rating_view")

    val max = Util.spark.sql("select max(rating) from rating_view")
    max.show()

    val min = Util.spark.sql("select min(rating) from rating_view")
    min.show()

    Util.spark.stop()
  }
}
