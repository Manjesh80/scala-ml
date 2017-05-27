name := "scala-ml"

version := "1.0"

//libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.1"


val sparkVersion = "2.0.0"


libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12",

  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,

  "org.scalatest" % "scalatest_2.11" % "3.0.1"
)


resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
  "Apache Repository" at "https://repository.apache.org/content/repositories/releases"
)


scalaVersion := "2.11.7"


