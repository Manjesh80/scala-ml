package com.manjesh.sparkml.maths

import org.scalatest._
import com.manjesh.sparkml.maths._


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

}
