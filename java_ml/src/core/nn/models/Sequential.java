package core.nn.models;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

import core.math.linalg.Matrix;
import core.nn.Activation;
import core.nn.layers.ActivationLayer;
import core.nn.layers.DenseLayer;
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

    public void saveModel(String path) {
        try {
            FileOutputStream fileOut = new FileOutputStream(path);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this);
            out.close();
            fileOut.close();
            System.out.printf("Serialized Sequential model is saved in " + path + "\n");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Sequential loadModel(String path) {
        try {
            FileInputStream fileIn = new FileInputStream(path);
            ObjectInputStream in = new ObjectInputStream(fileIn);

            Sequential model = (Sequential) in.readObject();

            in.close();
            fileIn.close();

            System.out.println("Deserialized Sequential model from " + path + "\n");
            return model;

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error loading model from " + path);
            return new Sequential();
        }
    }

    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new DenseLayer(2, 3));
        model.add(new ActivationLayer(Activation.Sigmoid));
        System.out.println(model);
        model.saveModel("model.model");
        Sequential model2 = Sequential.loadModel("model.model");
        System.out.println(model2);
    }

    @Override
    public void Training() {
        for (Layer layer : layers) {
            layer.setTraining(true);
        }
    }

    @Override
    public void Evaluation() {
        for (Layer layer : layers) {
            layer.setTraining(false);
        }
    }

    @Override
    public ArrayList<Layer> getTrainableLayers() {
        ArrayList<Layer> trainableLayers = new ArrayList<>();
        for (Layer layer : layers) {
            if (layer.isTrainable()) {
                trainableLayers.add(layer);
            }
        }
        return trainableLayers;
    }
}
