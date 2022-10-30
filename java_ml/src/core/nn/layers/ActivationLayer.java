package core.nn.layers;

import core.math.linalg.Matrix;
import core.nn.Activation;

public class ActivationLayer extends Layer {
    private Activation activation;

    public ActivationLayer(Activation activation) {
        this.activation = activation;
    }

    public Matrix forward(Matrix input) {
        return activation.f(input);
    }

    public Matrix backward(Matrix input, Matrix gradOutput) {
        return activation.df(input).elementWiseProduct(gradOutput);
    }

}
