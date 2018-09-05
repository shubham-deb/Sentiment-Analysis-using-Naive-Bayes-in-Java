package sentiment_analysis;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.apache.mahout.vectorizer.TFIDF;
 
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
 
public class NaiveBayes {
 
	static Configuration configuration = new Configuration();
	//public NaiveBayes() {
	//	configuration.set("mapred.job.tracker", "[server]:9001");
	//	configuration.set("fs.file.impl", "com.conga.services.hadoop.patch.HADOOP_7682.WinLocalFileSystem");
	
	//}
 
	String inputFilePath = "input/tweets.txt";
	String sequenceFilePath = "input/tweets-seq";
	String labelIndexPath = "input/labelindex";
	String vectorsPath = "input/tweets-vectors";
	String modelPath = "input/model";
	 
	public static void main(String[] args) throws Throwable {
		NaiveBayes nb = new NaiveBayes();
		nb.configuration.set("fs.file.impl", "com.conga.services.hadoop.patch.HADOOP_7682.WinLocalFileSystem");
		nb.inputDataToSequenceFile();
		nb.sequenceFileToSparseVector();
		nb.trainNaiveBayesModel();
		// Classify new tweetss after training
		Scanner scan = new Scanner(System.in);
		TweetClassifier tc = new TweetClassifier();
		while(true)
		{
			System.out.print("Please enter a tweet: ");
			System.out.println();
			String input = scan.nextLine();
			tc.classifyNewTweet(input,configuration);
		}
	}
 
	public void inputDataToSequenceFile() throws Exception {
		BufferedReader reader = new BufferedReader(
				new FileReader(inputFilePath));
		FileSystem fs = FileSystem.getLocal(configuration);
		Path seqFilePath = new Path(sequenceFilePath);
		fs.delete(seqFilePath, false);
		SequenceFile.Writer writer = SequenceFile.createWriter(fs,
				configuration, seqFilePath, Text.class, Text.class);
		int count = 0;
		try {
			String line;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split("\t");
				writer.append(new Text("/" + tokens[0] + "/tweet" + count++),
						new Text(tokens[1]));
			}
		} finally {
			reader.close();
			writer.close();
		}
	}
 
	void sequenceFileToSparseVector() throws Exception {
		SparseVectorsFromSequenceFiles svfsf = new SparseVectorsFromSequenceFiles();
		svfsf.run(new String[] { "-i", sequenceFilePath, "-o", vectorsPath,
				"-ow" });
	}
 
	void trainNaiveBayesModel() throws Exception {
		TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
		trainNaiveBayes.setConf(configuration);
		trainNaiveBayes.run(new String[] { "-i",
				vectorsPath + "/tfidf-vectors", "-o", modelPath, "-li",
				labelIndexPath, "-el", "-c", "-ow" });
	}
 
	
 
	
}