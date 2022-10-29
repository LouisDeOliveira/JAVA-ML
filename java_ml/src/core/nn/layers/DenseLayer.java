package core.nn.layers;

import java.io.Serializable;

import core.math.linalg.Matrix;
import core.nn.*;

public class DenseLayer extends Layer implements Serializable {
    private Matrix weights;
    private Matrix biases;
    private Activation activation;
    private int inputSize;
    private int outputSize;

    public DenseLayer(int inputSize, int outputSize, Activation activationFunction) {
        ;
        this.activation = activationFunction;
        this.weights = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { inputSize, outputSize }));
        this.biases = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { outputSize, 1 }));
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    @Override
    public Matrix forward(Matrix input) {
        Matrix output = weights.transposed().dot(input);
        output.add(biases);
        return activation.f(output);
    }

    public Activation getActivation() {
        return activation;
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    public void setBiases(Matrix biases) {
        this.biases = biases;
    }

    @Override
    public String toString() {
        return "Dense Layer: " + inputSize + " -> " + outputSize + " " + activation.toString();
    }

}
