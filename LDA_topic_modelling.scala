var corpus = sc.wholeTextFiles("file:///vagrant/opinion_leaders/*").map(_._2).map(_.toLowerCase())
corpus.count()
corpus.takeSample(false, 1)
val corpus_df = corpus.zipWithIndex.toDF("corpus", "id")
import org.apache.spark.ml.feature.RegexTokenizer
val tokenizer = new RegexTokenizer() .setPattern("[\\W_]+") .setMinTokenLength(4)  .setInputCol("corpus") .setOutputCol("tokens")
val tokenized_df = tokenizer.transform(corpus_df)
import org.apache.spark.ml.feature.StopWordsRemover
val stopWordsRemover = new StopWordsRemover() .setInputCol("tokens") .setOutputCol("filtered")
val filtered_df = stopWordsRemover.transform(tokenized_df)
import org.apache.spark.ml.feature.CountVectorizer
val vectorizer = new CountVectorizer() .setInputCol("filtered") .setOutputCol("features") .setVocabSize(10000).setMinDF(5).fit(filtered_df)
val countVectors = vectorizer.transform(filtered_df).select("id", "features")
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row
val lda_countVector = countVectors.map { case Row(id: Long, countVector: Vector) => (id, countVector) }
val numTopics = 4
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
val lda = new LDA() .setK(numTopics) .run(lda_countVector) .asInstanceOf[DistributedLDAModel]
val topicIndices = lda.describeTopics(maxTermsPerTopic = 100)
val vocabList = vectorizer.vocabulary
val topics = topicIndices.map { case (terms, termWeights) => terms.map(vocabList(_)).zip(termWeights)}
println(s"$numTopics topics:")
topics.zipWithIndex.foreach { case (topic, i) =>
  println(s"TOPIC $i")
  topic.foreach { case (term, weight) => println(s"$term\t$weight") }
  println(s"==========")
}
val termArray = topics.zipWithIndex
val termRDD = sc.parallelize(termArray)
val termRDD2 =termRDD.flatMap( (x: (Array[(String, Double)], Int)) => {
  val arrayOfTuple = x._1
  val topicId = x._2
  arrayOfTuple.map(el => s"${(el._1)}\t${(el._2)}\t${(topicId)}")
})
termRDD2.repartition(1).saveAsTextFile("file:///vagrant/myfile3")
val topicRDD = lda.topTopicsPerDocument(3).map {f => (f._1, f._2 zip f._3)}.map(f => s"${(f._1)}\t${f._2.map(k => k._1 + ":" + k._2).mkString(" ")}")
topicRDD.repartition(1).saveAsTextFile("file:///vagrant/myfile")

