package core.nn.layers;

import java.io.Serializable;

import core.math.linalg.Matrix;
import core.nn.Activation;

public class ActivationLayer extends Layer implements Serializable {
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

    public void applyGradient(Matrix input, Matrix gradOutput) {
        // No weights to update
    }

    @Override
    public String toString() {
        return "Activation Layer: " + activation.toString();
    }

    @Override
    public boolean isTrainable() {
        return false;
    }

}
