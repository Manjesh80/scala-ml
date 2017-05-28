package com.manjesh.sparkml.util

import org.apache.spark.sql.types.{StringType, StructField, StructType}

/**
  * Created by cloudera on 5/27/17.
  */
object ProcessingUtils {

  def getUserSchema: StructType = {

    return StructType(
      Array(
        StructField("no", StringType, true),
        StructField("age", StringType, true),
        StructField("gender", StringType, true),
        StructField("occupation", StringType, true),
        StructField("zipCode", StringType, true)
      )
    )

  }

}
