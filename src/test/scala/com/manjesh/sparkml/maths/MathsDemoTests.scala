package com.manjesh.sparkml.maths

import org.scalatest._
import com.manjesh.sparkml.maths._
import breeze.linalg.{normalize, _}
import breeze.stats.mean


/**
  * Created by cloudera on 5/26/17.
  */
class MathsDemoTests extends FlatSpec with Matchers {

  "A vector " should "give a correct value" in {
    val m = new MathsDemo()
    assert(m.returnTwo == 2)

  }

  "2 complete number" should "add properly" in {
    val k = new MathsDemo()
    println(k.addComplexNumbers)
    assert(k.addComplexNumbers === "4.0 + 2.0i")

  }

  "List complete number" should "add properly" in {
    val k = new MathsDemo()
    println("List result ==> " + k.addListOfComplexNumbers)
    assert(k.addListOfComplexNumbers === "9.0 + 3.0i")

  }

  "Update dense vector" should "updated properly" in {
    val k = new MathsDemo()
    println("Dense vector result ==> " + k.updateDesnseVector)
    assert(k.updateDesnseVector === "DenseVector(9.0, 4.0, 5.0, 6.0)")

  }

  "Create sparse vector and add " should "add properly" in {
    val k = new MathsDemo()
    println("Create sparse vector and add ==> " + k.createSparseVectorAndAdd)
    assert(k.createSparseVectorAndAdd === "SparseVector((0,2.0), (1,4.0), (4,6.0))")

  }

  "Create sparse vector from Spark " should "add properly" in {
    val k = new MathsDemo()
    println("Create sparse vector and add ==> " + k.createVectorFromSpark)
    println("argmax ==> " + k.createVectorFromSpark.argmax)
    println("numNonzeros ==> " + k.createVectorFromSpark.numNonzeros)
    println("size ==> " + k.createVectorFromSpark.size)
    println("toArray ==> " + k.createVectorFromSpark.toArray)
    println("toJson ==> " + k.createVectorFromSpark.toJson)


    assert(k.createVectorFromSpark.toString === "(4,[0,2,3],[1.0,2.0,3.0])")

  }

  "Vector addition " should "add properly" in {
    val k = new MathsDemo()
    println("Create sparse vector and add ==> " + k.vectorAddition)
    assert(k.vectorAddition.toString === "0.75")

  }

  "Mean of a vector " should "work properly" in {
    val meanValue = breeze.stats.mean(DenseVector(0.0, 1.5, 2.0))
    println(meanValue)
    assert(meanValue.toString === "1.1666666666666667")

  }


  /**
    *
    * Normalized vector: Every vector has a magnitude, which is calculated using the Pythagoras
    * theorem as |v| = sqrt(x^2 + y^2 + z^2); this magnitude is a length of a line from the origin
    * point (0,0,0) to the point indicated by the vector. A vector is normal if its magnitude is 1.
    * Normalizing a vector means changing it so that it points in the same direction (beginning
    * from the origin), but its magnitude is one. Hence, a normalized vector is a vector in the
    * same direction, but with norm (length) 1. It is denoted by ^X and is given by the following
    * formula:
    *
    *
    */
  "Normalized vector " should "work properly" in {

    val vector = DenseVector(3.0, 4.0)
    println("Normiazlized vector " + normalize(vector))
    println("Norm vector " + norm(vector))
    assert(1 === 1)

    val v = DenseVector(-0.4326, -1.6656, 0.1253, 0.2877, -1.1465)
    val nm = norm(v, 1)
    println(nm)
    val nmlize = normalize(v)
    println(norm(nmlize))
  }

  "Compare operations" should "work fine" in {

    val a1 = DenseVector(1.0, 2.0, 3.0)
    val b1 = DenseVector(1.0, 4.0, 9.0)
    println(a1 :== b1)
    println((a1 :<= b1))
    println((a1 :>= b1))
    println((a1 :< b1))
    println((a1 :> b1))
    assert(1 === 1)

  }

}
