import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;

public class LDATrainModel {

	/*
	 * Model for LDA
	 * Contains all relevant information about the current run
	 */
	
	/*------------------------------------------------------------*/
	//The model variables
	/*------------------------------------------------------------*/
	// store the data object => contains dictionary
	public LDAData data;
	
	// No.of Documents
	public int M;
	// Vocabulary Size
	public int V;
	// Number of topics
	public int K;
	
	//hyper-parameters for smoothing
	public double alpha, beta;
	
	//number of iterations to be run
	public int numIters;
	
	// lda parameters to be estimated
	// theta - (document topic distribution : size M * K)
	public double[][] theta;
	// phi - (topic word distribution : size K * V)
	public double[][] phi;


	/*--------------------------------------------------------------*/
	// Variables useful while sampling and running Lda
	/*--------------------------------------------------------------*/
	// the main model variables used for each gibbs sampling
	public Vector<Integer>[] currentTopicAss;
	// nw[i][j] : size V * K : denotes number of times word i is assigned topic j 
	public int[][] nw;
	// nd[i][j] : size M * K : denotes number of words in document i assigned to topic j
	public int[][] nd;
	// nwSum[j] : size K : total number of words assigned to topic j
	public int[] nwSum;
	// ndSum : size M : total number of words in document
	public int[] ndSum;
	
	// contains probabilities used for sampling topic to word
	public double[] samplingProb;
	
	
	/*--------------------------------------------------------------*/
	// Constructor
	/*--------------------------------------------------------------*/
	
	//constructor for setting default values
	public LDATrainModel(){
		
		setDefaultValues();
	}
	
	//customized model setting => prefer the above function
	public LDATrainModel(int K, double alpha, double beta, int numIters){
		
		setDefaultValues();
		
		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.numIters = numIters;
		
	}
	
	/*--------------------------------------------------------------*/
	// model parameters setter functions
	/*--------------------------------------------------------------*/
	
	//sets the default values for the model
	public void setDefaultValues(){
		
		M = 0;
		V = 0;
		K = 100;
		alpha = 50.0/K;
		beta = 0.1;
		numIters = 2000;
		
		currentTopicAss = null;
		nw = null;
		nd = null;
		nwSum = null;
		ndSum = null;
		
		theta = null;
		phi = null;
		
	}
	
	//read file to get parameters
	public boolean setParamsFromFile(String configFile){
		
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(configFile));
			String line;
			while( (line = br.readLine()) != null){
				
				StringTokenizer tokenizer = new StringTokenizer(line , "= \t\r\n");
				
				int noOfTokens = tokenizer.countTokens();
				if(noOfTokens != 2){
					continue;
				}
				
				String option = tokenizer.nextToken();
				String value = tokenizer.nextToken();
				
				if( option.equalsIgnoreCase("alpha")){
					alpha = Double.parseDouble(value);
				}
				if( option.equalsIgnoreCase("beta")){
					beta = Double.parseDouble(value);
				}
				if( option.equalsIgnoreCase("topics")){
					K = Integer.parseInt(value);
				}
				if( option.equalsIgnoreCase("iters")){
					numIters = Integer.parseInt(value);
				}
				
			}
			br.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	
	/*-------------------------------------------------------------*/
	// Data Loading and model creation functions
	/*--------------------------------------------------------------*/
	
	//load the ldaData
	public void loadModel(){
		
		/* This call read all the input files and creates dictionary
			else if already read, returns the read data.
			
			It is duty of config file to ensure same features in train and test
		*/
		//0 => for training
		//1 => for testing
		data = new LDAData();
		try {
			data.createData("train.txt");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	//once the model is loaded, it initializes the parameters of the model
	//for training
	public void initModel(){
		
		samplingProb = new double[K];
		
		//loads the training model
		loadModel();
		M = data.M;
		V = data.V;
		
		// value of K taken from configFile or default value
		int wordIter,topicIter,docIter;
		
		nw = new int[V][K];
		for(wordIter = 0; wordIter < V; wordIter++){
			for(topicIter = 0; topicIter < K; topicIter++){
				nw[wordIter][topicIter] = 0;
			}
		}
		
		nd = new int[M][K];
		for(docIter = 0; docIter < M; docIter++){
			for(topicIter = 0; topicIter < K; topicIter++){
				nd[docIter][topicIter] = 0;
			}
		}
		
		nwSum = new int[K];
		for(topicIter = 0; topicIter < K; topicIter++){
			nwSum[topicIter] = 0;
		}
		
		ndSum = new int[M];
		for(docIter = 0; docIter < M; docIter++){
			ndSum[docIter] = 0;
		}
		
		//used to store all the current topic assignments
		currentTopicAss = new Vector[M];
		int docLength,n,topic;
		
		for(docIter = 0; docIter < M; docIter++){
			
			//read each document length and create a vector
			docLength = data.docs[docIter].length;
			currentTopicAss[docIter] = new Vector<Integer>();
			
			//initialize by a random topic assignment
			for(n=0; n< docLength; n++){
				
				//add randomly sampled point
				topic = (int)Math.floor(Math.random()*K);
				currentTopicAss[docIter].add(topic);
				
				//number of instances of word assigned to topic j
				nw[data.docs[docIter].words[n]][topic] += 1;
				//number of words in document i assigned top topic j
				nd[docIter][topic] += 1;
				//total number of words assigned to topic j
				nwSum[topic] += 1;
			}
			//total number of words in ith document
			ndSum[docIter] = docLength;
			
		}
		
		theta = new double[M][K];
		phi = new double[K][V];
		
	}
	
}
