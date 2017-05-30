name := "scala-ml"

version := "1.0"

//libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.1"


val sparkVersion = "2.0.0"

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
  "Apache Repository" at "https://repository.apache.org/content/repositories/releases"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.scalatest" % "scalatest_2.11" % "3.0.1"
)

libraryDependencies += "org.jfree" % "jfreechart" % "1.0.19"
libraryDependencies += "com.github.wookietreiber" %% "scala-chart" % "latest.integration"


scalaVersion := "2.11.7"


