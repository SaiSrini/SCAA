
/*
This file will perform estimation

Author: K. Sai Srinivas
Project: SCAA
Date: 29th April, 2016.
*/

public class LDAEstimator {

	//------------------------------------------------------
	//The variables
	//------------------------------------------------------
	protected LDATrainModel trainModel;
	
	//------------------------------------------------------
	//Constructor
	//------------------------------------------------------
	/***************************/
	//method to initialize the LDAEstimator
	public LDAEstimator(){
		trainModel = new LDATrainModel();
		trainModel.initModel();
		
	}
	
	//------------------------------------------------------
	//Methods
	//------------------------------------------------------
	/***************************/
	//This method will estimate values by sampling
	public void Estimate(){
		//Run the estimator for numIter number of iterations
		
		for(int i=0; i< trainModel.numIters; i++){
			
			System.out.printf("The iteration number is %d...\n", i);
			//For each document in the corpus
			for(int j =0; j< trainModel.M; j++){
				//for each word in the document
				for(int k=0; k< trainModel.data.docs[i].length; k++){
					//z = z[j][k]
					//sample from p(z_i | z_-i, w)
					int topic = Sample(j,k);
					trainModel.currentTopicAss[j].set(k, topic);
				}
			}
		}
		
		//after completing the sampling
		System.out.println("Gibbs Sampling Completed. Huzzah!");
		
		//compute the values of Thetha and Phi
		ComputeTheta();
		ComputePhi();
	}
	
	/***************************/
	//This is the sampling function that finds the topic for the next iteration
	public int Sample(int j, int k){
		//variables
		int topic;	//The current topic
		int word;	//The word
		double temp;	//used in sampling
		
		topic = trainModel.currentTopicAss[j].get(k);
		word = trainModel.data.docs[j].words[k];
		
		//Because we are removing the topic assignment, we decrement arrays accordingly
		trainModel.nw[word][topic]--;
		trainModel.nd[j][topic]--;
		trainModel.nwSum[topic]--;
		trainModel.ndSum[j]--;
		
		double wordBeta = trainModel.V * trainModel.beta;
		double topicAlpha = trainModel.K * trainModel.alpha;
		
		//calculate probabilities
		for(int i=0; i< trainModel.K; i++){
			temp = (trainModel.nd[j][i] + trainModel.alpha) / (trainModel.ndSum[j] + topicAlpha);
			trainModel.samplingProb[i] = temp * ((trainModel.nw[word][topic] + trainModel.beta) / (trainModel.nwSum[i] + wordBeta));
			
		}
		
		//Sample from the probabilities
		
		//Finding cumulative probabilities
		for(int i=1; i< trainModel.K; i++){
			trainModel.samplingProb[i] += trainModel.samplingProb[i-1];
		}
		//select a random number
		double selector = Math.random() * trainModel.samplingProb[trainModel.K - 1];
		for(topic = 0; topic <trainModel.K; topic++){
			if(trainModel.samplingProb[topic] > selector)
				break;
		}
		
		//The returned topic is the selected one
		trainModel.nw[word][topic]++;
		trainModel.nd[j][topic]++;
		trainModel.nwSum[topic]++;
		trainModel.ndSum[topic]++;
		
		return topic;
	}
	
	/***************************/
	//compute the values of Theta
	public void ComputeTheta(){
		
		double topicAlpha = trainModel.K * trainModel.alpha;
		
		
		for(int i=0; i< trainModel.M; i++){
			for(int j=0; j<trainModel.K; j++){
				trainModel.theta[i][j] = (trainModel.nd[i][j] + trainModel.alpha) / (trainModel.ndSum[i] + topicAlpha);
			}
		}
	}
	
	/***************************/
	//compute the value of Phi
	public void ComputePhi(){
		
		double wordBeta = trainModel.V * trainModel.beta;
		
		for(int i=0; i< trainModel.K; i++){
			for(int j = 0; j < trainModel.V; j++){
				trainModel.phi[i][j] = (trainModel.nw[j][i] + trainModel.beta) / (trainModel.nwSum[j] + wordBeta);
			}
		}
	}
}
