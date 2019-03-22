package p3.p3;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Scanner;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;
import weka.filters.Filter;

public class App 
{
	
    public static void main( String[] args ) throws Exception
    {
    	
    	ArffLoader loader = new ArffLoader();
    	File f = new File("C:\\Users\\hkimb\\eclipse-workspace\\p3\\src\\main\\java\\p3\\p3\\train.arff");
    	loader.setFile(f);
    	Instances train_instance = loader.getDataSet();
    	
    	ArffLoader loader2 = new ArffLoader();
    	File f2 = new File ("C:\\Users\\hkimb\\eclipse-workspace\\p3\\src\\main\\java\\p3\\p3\\eval.arff");
    	loader2.setFile(f2);
    	Instances test_instance = loader2.getDataSet();
    	

        train_instance.setClassIndex(train_instance.numAttributes() - 1);
        test_instance.setClassIndex(test_instance.numAttributes() - 1);
        
        SMO smo = new SMO();
        smo.buildClassifier(train_instance);

        Evaluation evaluation = new Evaluation(train_instance);
        evaluation.evaluateModel(smo, test_instance);
        

        File file = new File("C:\\Users\\hkimb\\eclipse-workspace\\p3\\src\\main\\java\\p3\\p3\\results.txt");
		PrintWriter writer = new PrintWriter(file, "UTF-8");
       
		FileInputStream ids = new FileInputStream("C:\\Users\\hkimb\\eclipse-workspace\\p3\\src\\main\\java\\p3\\p3\\ids.txt");
		Scanner scanner = new Scanner(ids, "UTF-8");
		
		
        writer.println("True positives: " + evaluation.numTruePositives(0) + "\tFalse positives: " + evaluation.numFalsePositives(0) + "\tTrue negatives: " + evaluation.numTrueNegatives(0) + "\tFalse negatives: " + evaluation.numFalseNegatives(0));
        writer.println("Precision: " + evaluation.precision(0) + "\tRecall: " + evaluation.recall(0) + "\tF1-score: " + (2.0*evaluation.precision(0)*evaluation.recall(0))/(evaluation.precision(0) + evaluation.recall(0)));
        
        Instances unlabeled = DataSource.read("C:\\Users\\hkimb\\eclipse-workspace\\p3\\src\\main\\java\\p3\\p3\\eval.arff");
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        // create copy
        Instances labeled = new Instances(unlabeled);
        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
        	double clsLabel = smo.classifyInstance(unlabeled.instance(i));
        	labeled.instance(i).setClassValue(clsLabel);
        	writer.print(scanner.nextLine());
        	if (labeled.instance(i).classValue() == 0) writer.println("\tRelevant");
        	else writer.println("\tIrrelevant");
        }
        
        scanner.close();
        writer.close();
        
    }
}
