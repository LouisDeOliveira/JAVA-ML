package core.nn.layers;

import core.math.linalg.Matrix;
import core.nn.*;

public class DenseLayer extends Layer {
    private Matrix weights;
    private Matrix biases;
    private int inputSize;
    private int outputSize;
    private Activation activation;

    public DenseLayer(int inputSize, int outputSize, Activation activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        this.weights = new Matrix(Initializer.NormalInitializer(0, 1).initialize(new int[] { inputSize, outputSize }));
        this.biases = new Matrix(Initializer.NormalInitializer(0, 1).initialize(new int[] { outputSize, 1 }));
    }

    @Override
    public Matrix forward(Matrix input) {
        Matrix output = weights.transposed().dot(input);
        output.add(biases);

        return activation.f(output);
    }

    public static void main(String[] args) {
        DenseLayer layer = new DenseLayer(2, 3, Activation.ReLU);
        Matrix input = new Matrix(new double[][] { { 1 }, { 3 } });
        Matrix output = layer.forward(input);
        System.out.println(output);
    }

}
