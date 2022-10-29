package core.nn.models;

import java.io.Serializable;
import java.util.ArrayList;

import core.math.linalg.Matrix;
import core.nn.layers.Layer;

public class Sequential extends Model implements Serializable {
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

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    @Override
    public String toString() {
        String output = "Sequential Model: \n";
        for (Layer layer : layers) {
            output += layer.toString() + "\n";
        }

        return output;
    }
}
