package core.nn.layers;

import java.io.Serializable;

import core.math.linalg.Matrix;
import core.nn.Initializer;

public class DropoutLayer extends Layer implements Serializable {
    private float p;
    private boolean training;

    public DropoutLayer(float p) {
        this.p = p;
    }

    @Override
    public Matrix forward(Matrix input) {
        if (training) {
            Matrix mask = new Matrix(Initializer.BinaryInitializer(p).initialize(input.getSize()));
            double rate = 1 - mask.sum() / (mask.getSize()[0] * mask.getSize()[1]);
            return input.elementWiseProduct(mask).scale((double) 1 / (1 - rate));
        } else {
            return input;
        }
    }

    @Override
    public Matrix backward(Matrix input, Matrix gradOutput) {
        return gradOutput;
    }

    @Override
    public void applyGradient(Matrix input, Matrix gradOutput) {
        // No weights to update
    }

    @Override
    public boolean isTrainable() {
        return false;
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }

    @Override
    public String toString() {
        return "Dropout Layer: p = " + p;
    }

    public static void main(String[] args) {
        Matrix input = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        DropoutLayer layer = new DropoutLayer(0.1f);
        layer.setTraining(true);
        for (int i = 0; i < 10; i++) {
            System.out.println(input.sum());
            Matrix output = layer.forward(input);
            System.out.println(output.sum());

        }
    }

}
