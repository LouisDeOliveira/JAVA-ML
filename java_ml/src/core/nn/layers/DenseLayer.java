package core.nn.layers;

import core.math.linalg.Matrix;
import core.nn.*;

public class DenseLayer extends Layer {
    private Matrix weights;
    private Matrix biases;
    private Activation activation;

    public DenseLayer(int inputSize, int outputSize, Activation activationFunction) {
        ;
        this.activation = activationFunction;
        this.weights = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { inputSize, outputSize }));
        this.biases = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { outputSize, 1 }));
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
        return "biases= \n" + biases + ",\n weights= \n" + weights;
    }

}
