
public class LDAMain {

	
	
	double[][] hellDistance;
	
	public static void main(String args[]){
		
		//
		// Used to estimates training topic assignments  
		//
		LDAEstimator est = new LDAEstimator();
		est.Estimate();
		
		//
		// Performs inference on test model
		//
		LDAInferencer inf = new LDAInferencer(est.trainModel);
		inf.inference();
		
		LDAMain main = new LDAMain();
		main.computeHellingerDistance(est.trainModel, inf.testModel);
		
		System.out.println(main.hellDistance);
	}
	
	public void computeHellingerDistance(LDATrainModel trainModel, LDATestModel testModel){
		
		//first set the size of the hell distance matrix
		// size: testNoofDocs * trainNoofDocs
		hellDistance = new double[testModel.M][trainModel.M];
		
		//now compute the hellinger distance for all the pairs
		for(int i = 0; i < testModel.M; i++){
			for(int j = 0; j < trainModel.M; j++){
				
				hellDistance[i][j] = 0;
				for(int k =0; k < trainModel.K; k++ ){
					hellDistance[i][j] +=  Math.pow(( Math.sqrt(testModel.theta[i][k]) - Math.sqrt(trainModel.theta[j][k]) ), 2);
				}
				hellDistance[i][j] = hellDistance[i][j]/2.0;
				hellDistance[i][j] = Math.sqrt(hellDistance[i][j]);
			}
		}
	}
}
