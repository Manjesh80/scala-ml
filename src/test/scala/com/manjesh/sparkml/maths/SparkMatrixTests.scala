package com.manjesh.sparkml.maths

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest._
import org.apache.log4j._
import org.apache.spark.rdd.RDD

/**
  * Created by cloudera on 5/27/17.
  */
class SparkMatrixTestsSuite extends FlatSpec with Matchers {

  "Spark Matrix" should "work fine " in {
    (new SparkMatrixTests()).testSparkMatrix()
  }

}

class SparkMatrixTests {
  def testSparkMatrix() = {

    val spkConfig = (new SparkConf()).
      setMaster("local").
      setAppName("Spark Matrix tests");

    val sc = new SparkContext(spkConfig)
    sc.setLogLevel("ERROR")

    val denseData = Seq(
      Vectors.dense(0.0, 1.0, 2.1),
      Vectors.dense(3.0, 2.0, 4.0),
      Vectors.dense(5.0, 7.0, 8.0),
      Vectors.dense(9.0, 0.0, 1.1)
    )
    val sparseData = Seq(
      Vectors.sparse(3, Seq((1, 1.0), (2, 2.1))),
      Vectors.sparse(3, Seq((0, 3.0), (1, 2.0), (2, 4.0))),
      Vectors.sparse(3, Seq((0, 5.0), (1, 7.0), (2, 8.0))),
      Vectors.sparse(3, Seq((0, 9.0), (2, 1.0)))
    )

    val denseMat = new RowMatrix(sc.parallelize(denseData, 2))
    val sparseMat = new RowMatrix(sc.parallelize(sparseData, 2))

    println("Dense Matrix - Num of Rows :" + denseMat.numRows())
    println("Dense Matrix - Num of Cols:" + denseMat.numCols())
    println("Sparse Matrix - Num of Rows :" + sparseMat.numRows())
    println("Sparse Matrix - Num of Cols:" + sparseMat.numCols())


    println("Show indexed Matrix")

    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0))
    ).map(x => IndexedRow(x._1, x._2))
    val indexedRows: RDD[IndexedRow] = sc.parallelize(data, 2)
    val indexedRowsMat = new IndexedRowMatrix(indexedRows)
    println("Indexed Row Matrix - No of Rows: " +
      indexedRowsMat.numRows())
    println("Indexed Row Matrix - No of Cols: " +
      indexedRowsMat.numCols())


    println("Showing Coordinated matrix")

    val entries = sc.parallelize(Seq(
      (0, 0, 1.0),
      (0, 1, 2.0),
      (1, 1, 3.0),
      (1, 2, 4.0),
      (2, 2, 5.0),
      (2, 3, 6.0),
      (3, 0, 7.0),
      (3, 3, 8.0),
      (4, 1, 9.0)), 3).map { case (i, j, value) =>
      MatrixEntry(i, j, value)
    }
    val coordinateMat = new CoordinateMatrix(entries)
    println("Coordinate Matrix - No of Rows: " +
      coordinateMat.numRows())
    println("Coordinate Matrix - No of Cols: " +
      coordinateMat.numCols())



    sc.stop()

  }
}
