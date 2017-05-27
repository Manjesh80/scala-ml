package com.manjesh.sparkml.maths

/**
  * Created by cloudera on 5/26/17.
  */

import breeze.linalg.{DenseVector, SparseVector}
import breeze.math.Complex


class MathsDemo {

  def returnTwo: Int = {
    return 2
  }

  def addComplexNumbers: String = {

    val i = Complex.i
    println(((1 + 2 * i) + (1 + 2 * 1)).toString())
    return ((1 + 2 * i) + (1 + 2 * 1)).toString()
  }

  def addListOfComplexNumbers = {
    val i = Complex.i
    (List((1 + 1 * i), (3 + 1 * i), (5 + 1 * i))).sum.toString()
  }

  def updateDesnseVector = {

    val dv = DenseVector(2f, 4f, 5f, 6f)
    dv.update(0, 9f)
    println(dv)
    dv.toString()
  }

  def createSparseVectorAndAdd = {
    val sv: SparseVector[Double] = SparseVector(5)()
    sv(0) = 1
    sv(1) = 3
    sv(4) = 5

    val m: SparseVector[Double] = sv.mapActivePairs((i, x) => x + 1)
    println(m)
    m.toString()
  }

  def createVectorFromSpark = {
    val spkVector: org.apache.spark.mllib.linalg.Vector =
      org.apache.spark.mllib.linalg.Vectors.sparse(4, Seq((0, 1.0), (2, 2.0),
        (3, 3.0)));
    spkVector
  }

  def vectorAddition = {
    val a = DenseVector(0.5, 0.5, 0.5)
    val b = DenseVector(0.5, 0.5, 0.5)

    a dot b

  }

}
