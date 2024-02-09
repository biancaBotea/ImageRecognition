package c2b.run2;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;


public class App {
	static int step = 3;
	static int patch_size = 15;
	
	public static void main(String[] args) {
//		public static int STEP = 4;
		try {
			System.out.println("I started doing stuff!");
			VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>("/Users/bianca/CompVision", ImageUtilities.FIMAGE_READER);

//			ArrayList<float[]> testArray = getVectors(dataset.getRandomInstance(), 8, 8, 4);
//			System.out.print(testArray);
			
			int nTraining = 50; //change back to 80 later
			int nTesting = 20; //and 20
			
			GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(dataset, nTraining, 0, nTesting);
			GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			
			//creating the bag of words
			HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 30));
			System.out.println("Assigner done");
			
			FeatureExtractor<DoubleFV, FImage> extractor = new PatchExtractor(assigner);
			System.out.println("Extractor done");
	
			//automatically creates 15 one-vs-all classifiers
			System.out.println("Creating the classifiers");
			LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
//			LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTILABEL, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
			System.out.println("Starting trainig");
			ann.train(training);
			System.out.println("Traing done");
//			
//			System.out.println("Classify training");
//			for (final Entry<String, ListDataset<FImage>> entry : training.entrySet()) {
//				
//				for(FImage img : entry.getValue()) {
//					System.out.println(ann.classify(img));
//				}
//				
//			}
//			System.out.println("Classify testing");
//			for (final Entry<String, ListDataset<FImage>> entry : testing.entrySet()) {
//
//				for(FImage img : entry.getValue()) {
//					System.out.println(ann.classify(img));
//				}
//				
//			}
//			
			System.out.println("Evaluatino time!");
			ClassificationEvaluator<CMResult<String>, String, FImage> evaluator =
		            new ClassificationEvaluator<CMResult<String>, String, FImage>(ann, testing, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

	        Map<FImage, ClassificationResult<String>> guesses = evaluator.evaluate();
	        CMResult<String> result = evaluator.analyse(guesses);
	        System.out.println(result);
			
//			
		} catch(Exception e){
			e.printStackTrace();
		}
			
	}
	
	/**
	 * Gets the  feature vectors of the patches
	 * @param image : The image that we gest the patches from
	 * @param patch_size : the size of the patch, our patches are square so there is only one value needed.
	 * @param step : the step size
	 * @return
	 */
	public static List<LocalFeature<SpatialLocation, FloatFV>> getVector(FImage image, float patch_size, float step){
		
		List<LocalFeature<SpatialLocation, FloatFV>> allFeatures = new ArrayList<LocalFeature<SpatialLocation,FloatFV>>();
		
		RectangleSampler rectangles = new RectangleSampler(image, step, step, patch_size, patch_size);
		
		for(Rectangle rectangle : rectangles) {
			FImage patch = image.extractROI(rectangle);
			
			//flattens patch into vector array 
			float[] flattend = patch.getFloatPixelVector();
			
			//gets feature vectore
			FloatFV features = new FloatFV(flattend);
			SpatialLocation pos = new SpatialLocation(rectangle.x, rectangle.y);
			allFeatures.add(new LocalFeatureImpl<SpatialLocation, FloatFV>(pos,features));
		}
		
		return allFeatures;
	}
	
	
	/**
	 * Makes an assigner based on kMeans
	 * @param groupedDataset The training data set
	 * @return
	 */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset){
		
		List<float[]> allKeys = new ArrayList<float[]>();
		for (FImage img : groupedDataset) {
			
			List<LocalFeature<SpatialLocation, FloatFV>> fv = getVector(img, patch_size, step);
			for(LocalFeature<SpatialLocation, FloatFV> f : fv) {
				
				allKeys.add(f.getFeatureVector().values);
			}
		}

		//Performing k-means clustering on 500 classes
		FloatKMeans kMeans = FloatKMeans.createKDTreeEnsemble(150); // NEEDS TO BE CHANGE BACK TO 500
		System.out.println("Clustering time!");
		FloatCentroidsResult result = kMeans.cluster(allKeys.toArray(new float[][]{}));
		
		return result.defaultHardAssigner();
	}
	
}
class PatchExtractor implements FeatureExtractor<DoubleFV, FImage> {
	HardAssigner<float[], float[], IntFloatPair> assigner;
	
	public PatchExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
		
		this.assigner = assigner;
	}
	
	/**
	 * Extracts the features of the given image using the given assigner
	 * @param object The image we want to extract the features from
	 */
	public DoubleFV extractFeature(FImage object) {
		
		FImage image = object.getImage();
		
		BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
		
		BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
		
		new App();
		return spatial.aggregate(App.getVector(image,App.patch_size,App.step), image.getBounds()).normaliseFV();
	}
}


