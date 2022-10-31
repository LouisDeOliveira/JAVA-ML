package core.nn.layers;

import java.io.Serializable;

import core.math.linalg.Matrix;
import core.nn.*;

public class DenseLayer extends Layer implements Serializable {
    private Matrix weights;
    private Matrix biases;
    private int inputSize;
    private int outputSize;
    private boolean trainable;
    private boolean training;

    public DenseLayer(int inputSize, int outputSize) {
        ;
        this.weights = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { inputSize, outputSize }));
        this.biases = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { outputSize, 1 }));
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.trainable = true;
    }

    public DenseLayer(int inputSize, int outputSize, boolean trainable) {
        ;
        this.weights = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { inputSize, outputSize }));
        this.biases = new Matrix(
                Initializer.NormalInitializer(0, 1)
                        .initialize(new int[] { outputSize, 1 }));
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.trainable = trainable;
    }

    @Override
    public Matrix forward(Matrix input) {
        Matrix output = weights.transposed().dot(input);
        output.add(biases);
        return output;
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
        return "Dense Layer: " + inputSize + " -> " + outputSize;
    }

    @Override
    public Matrix backward(Matrix input, Matrix gradOutput) {
        return weights.dot(gradOutput);
    }

    @Override
    public void applyGradient(Matrix input, Matrix gradient) {
        biases = biases.substract(gradient);
        weights = weights.substract(input.dot(gradient.transposed()));

    }

    @Override
    public boolean isTrainable() {
        return trainable;
    }

    public void setTrainable(boolean trainable) {
        this.trainable = trainable;
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }

}
