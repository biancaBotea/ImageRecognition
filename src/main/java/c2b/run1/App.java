package c2b.run1;

import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
        // Temporary way of getting the images. Comment this out and add your own line if you want to change the path to
        // the training image dataset
        VFSGroupDataset<FImage> images = new VFSGroupDataset<FImage>("C:\\Users\\yan_b\\Documents\\Projects\\the-odin-project\\CompVision\\training", ImageUtilities.FIMAGE_READER);
        int nTraining = 90;
        int nTesting = 10;
        int photoSize = 16;

        for (final Map.Entry<String, VFSListDataset<FImage>> entry : images.entrySet()) {
            for (FImage fImage : entry.getValue()) {
                fImage = fImage.extractCenter(fImage.width,fImage.height);
                fImage.processInplace(new ResizeProcessor(photoSize,photoSize));
                fImage.normalise();
            }
        }

        GroupedRandomSplitter<String, FImage> splits =
                new GroupedRandomSplitter<String, FImage>(images, nTraining, 0, nTesting);

        GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
    }
}
