name := "app"
version := "1.0.0"
val sparkVersion="3.1.2"
libraryDependencies ++= Seq(
"org.apache.spark" %% "spark-core" % sparkVersion,
"org.apache.spark" %% "spark-sql" % sparkVersion,
"org.apache.spark" %% "spark-mllib" % sparkVersion
)