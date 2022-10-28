package core.nn.layers;

import core.math.linalg.Matrix;
import core.nn.*;

public class DenseLayer extends Layer {
    private Matrix weights;
    private Matrix biases;
    private int inputSize;
    private int outputSize;
    private Activation activation;
    private Matrix activationOutput;

    public DenseLayer(int inputSize, int outputSize, Activation activationFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
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
        activationOutput = output;

        return activation.f(output);
    }

    public Matrix getActivationValues() {
        return activationOutput;
    }

    public Activation getActivation() {
        return activation;
    }

    public void backward(Matrix biasGradient, Matrix weightGradient) {
        this.weights.add(weightGradient.transposed());
        this.biases.add(biasGradient);
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

}
