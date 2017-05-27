package com.manjesh.sparkml.maths

/**
  * Created by cloudera on 5/27/17.
  */

import breeze.linalg._
import breeze.optimize._
import org.scalatest._
import com.manjesh.sparkml.maths._
import breeze.stats.mean

class MatrixOpertations extends FlatSpec with Matchers {

  "Matrix addition" should "work fine" in {

    val a = DenseMatrix((1, 2), (3, 4))
    val b = DenseMatrix((1, 2), (3, 4))


    val c = a + b

    println(a)
    println(b)
    println(c)
  }

  "Matrix multiplication" should "work fine" in {

    val a = DenseMatrix((1, 2), (3, 4))
    val b = DenseMatrix((1, 2), (3, 4))


    val c = a * b

    //Elementwise comparison
    val d = a :<= b

    println(a)
    println(" -------------- ")
    println(b)
    println(" -------------- ")
    println(c)
    println(" -------------- ")
    println(d)

    println(" -------------- ")
    println(a :> b)
  }

  "Eigen value demos " should "work" in {

    val dm = DenseMatrix(
      (9.0, 0.0, 0.0),
      (0.0, 25.0, 0.0),
      (0.0, 0.0, 36.0))

    val es = eigSym(dm)

    val lambda = es.eigenvalues

    val evs = es.eigenvectors

    println("Eigen Symbol is : " + es)

    println("lambda is : " + lambda)

    println("evs is : " + evs)


  }

}
