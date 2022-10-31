package core;

import core.data.LabelDataLoader;
import core.data.MatrixDataLoader;
import core.data.OneHotEncoder;
import core.math.linalg.Matrix;
import core.nn.models.Sequential;

public class Test {
    public static void main(String[] args) {

        MatrixDataLoader loader = new MatrixDataLoader();
        loader.readData("./java_ml/data/t10k-images.idx3-ubyte");
        LabelDataLoader labelLoader = new LabelDataLoader(
                10);
        labelLoader.readData("./java_ml/data/t10k-labels.idx1-ubyte");

        OneHotEncoder encoder = new OneHotEncoder(10);
        Sequential model = Sequential.loadModel("MNIST.model");
        model.Evaluation();

        int correct = 0;

        for (int i = 0; i < loader.length(); i++) {
            Matrix input = loader.getData(i);
            int label = encoder.decode(labelLoader.getData(i));
            int prediction = encoder.decode(model.forward(input));

            if (label == prediction) {
                correct++;
            }
            if (i % 1000 == 0) {
                System.out.println("Step: " + i);
                System.out.println("Current Accuracy: " + (double) correct / (i + 1));
            }
        }

        System.out.println("Accuracy: " + (double) correct / loader.length());

    }
}
