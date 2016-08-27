import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import com.sigopt.spark.SigOptCrossValidator

// Load training data in LIBSVM format.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(10)
  .run(training)

// Compute raw scores on the test set.
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println(s"Accuracy = $accuracy")

// Save and load model
model.save(sc, "target/tmp/scalaLogisticRegressionWithLBFGSModel")
val sameModel = LogisticRegressionModel.load(sc,
  "target/tmp/scalaLogisticRegressionWithLBFGSModel")

# import org.apache.spark.{SparkConf, SparkContext}
# import org.apache.spark.sql.{Row, SQLContext}
# import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
# import org.apache.spark.ml.evaluation.RegressionEvaluator
# import org.apache.spark.ml.regression.LinearRegression
# import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
# import org.apache.spark.mllib.util.MLUtils



object LogisticRegressionCV {
  def main(args: Array[String]): Unit ={
    var clientToken = args.headOption.getOrElse("FAKE_CLIENT_TOKEN")
    val conf = new SparkConf().setAppName("SigOpt Example").setMaster("local")
    val spark = new SparkContext(conf)
    val sqlContext = new SQLContext(spark)
    import sqlContext.implicits._

    val cv = new SigOptCrossValidator("123")
    val lr = new LinearRegression()
    cv.setEstimator(lr)
    cv.setNumFolds(5)
    cv.setNumIterations(10)
    cv.setClientToken(clientToken)
    cv.setEvaluator(new RegressionEvaluator())

    // If your experiment has already been created, you can just set the ID instead
    // cv.setExperimentId("4866")
    cv.createExperiment(
      "SigOpt CV Example",
      List(("elasticNetParam", 0.0, 1.0, "double"), ("regParam", 0.0, 1.0, "double"))
    )
    val training = MLUtils.loadLibSVMFile(spark, "examples/data/sample_linear_regression_data.txt").toDF()
    cv.fit(training)
    spark.stop()
  }
}
