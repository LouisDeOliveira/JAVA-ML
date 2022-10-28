package core.nn.models;

import java.util.ArrayList;

import core.math.linalg.Matrix;
import core.nn.Activation;
import core.nn.layers.DenseLayer;
import core.nn.layers.Layer;

public class Sequential extends Model {
    private ArrayList<Layer> layers;

    public Sequential() {
        this.layers = new ArrayList<>();
    }

    public Sequential(ArrayList<Layer> layers) {
        this.layers = layers;
    }

    public void add(Layer layer) {
        this.layers.add(layer);
    }

    @Override
    public Matrix forward(Matrix input) {
        Matrix output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new DenseLayer(2, 3, Activation.ReLU));
        model.add(new DenseLayer(3, 1, Activation.Sigmoid));
        Matrix input = new Matrix(new double[][] { { 1 }, { 3 } });
        Matrix output = model.forward(input);
        System.out.println(output);
    }

}
