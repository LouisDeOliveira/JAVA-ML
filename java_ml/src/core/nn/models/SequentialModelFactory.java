package core.nn.models;

import core.nn.Activation;
import core.nn.layers.ActivationLayer;
import core.nn.layers.DenseLayer;

public class SequentialModelFactory {
    public static Sequential createSequentialModel(int[] layerSizes, Activation[] activations) {
        Sequential model = new Sequential();

        int inputSize = layerSizes[0];
        for (int i = 1; i < layerSizes.length; i++) {
            int outputSize = layerSizes[i];
            model.add(new DenseLayer(inputSize, outputSize));
            model.add(new ActivationLayer(activations[i - 1]));
            inputSize = outputSize;
        }

        return model;
    }

    public static void main(String[] args) {
        Sequential model = createSequentialModel(new int[] { 784, 128, 10 },
                new Activation[] { Activation.ReLU, Activation.Sigmoid });
        System.out.println(model);
    }
}
