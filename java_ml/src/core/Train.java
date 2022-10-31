package core;

import core.data.LabelDataLoader;
import core.data.MatrixDataLoader;
import core.nn.RealActivation;
import core.nn.VectorActivation;
import core.nn.Loss;
import core.nn.models.Sequential;
import core.optim.SGD;
import core.nn.layers.ActivationLayer;
import core.nn.layers.DenseLayer;
import core.nn.layers.DropoutLayer;

public class Train {
    public static void main(String[] args) {
        MatrixDataLoader loader = new MatrixDataLoader();
        loader.readData("./java_ml/data/train-images.idx3-ubyte");
        LabelDataLoader labelLoader = new LabelDataLoader(10);
        labelLoader.readData("./java_ml/data/train-labels.idx1-ubyte");

        Sequential model = new Sequential();
        model.add(new DenseLayer(784, 128));
        model.add(new DropoutLayer(0.1f));
        model.add(new ActivationLayer(RealActivation.ReLU));
        model.add(new DenseLayer(128, 10));
        model.add(new ActivationLayer(VectorActivation.Softmax));
        model.Training();
        SGD optimizer = new SGD(model, 0.001, Loss.MSE, true);
        int n_epochs = 3;
        for (int i = 0; i < n_epochs; i++) {
            for (int j = 0; j < loader.length(); j++) {
                optimizer.step(loader.getData(j), labelLoader.getData(j));
                if (j % 1000 == 0) {
                    System.out.println("Epoch: " + i + " Step: " + j);
                }
            }
        }

        model.saveModel("MNIST_128_dropout_softmax.model");

    }
}
