package core.optim;

import java.util.ArrayList;

import core.math.linalg.Matrix;
import core.nn.Activation;
import core.nn.Loss;
import core.nn.models.Model;
import core.nn.models.Sequential;
import core.nn.layers.*;

public class SGD {
    private double learning_rate;
    private Model model;
    private Loss loss;
    private boolean verbose;
    private ArrayList<Matrix> outputs;
    private ArrayList<Matrix> activations;
    private ArrayList<Matrix> gradients;

    public SGD(Model model, double learning_rate, Loss loss) {
        this.model = model;
        this.learning_rate = learning_rate;
        this.loss = loss;
        this.verbose = false;
        this.outputs = new ArrayList<Matrix>();
        this.activations = new ArrayList<Matrix>();
        this.gradients = new ArrayList<Matrix>();
    }

    public SGD(Model model, double learning_rate, Loss loss, boolean verbose) {
        this.model = model;
        this.learning_rate = learning_rate;
        this.loss = loss;
        this.verbose = verbose;
        this.outputs = new ArrayList<Matrix>();
        this.activations = new ArrayList<Matrix>();
        this.gradients = new ArrayList<Matrix>();
    }

    public void forwardPropagation(Matrix input) {
        outputs = new ArrayList<Matrix>();
        activations = new ArrayList<Matrix>();

        outputs.add(input);
        activations.add(input);

        for (int i = 0; i < model.getLayers().size(); i++) {
            Layer layer = model.getLayers().get(i);
            Matrix output = layer.getWeights().transposed().dot(activations.get(i));
            output.add(layer.getBiases());
            outputs.add(output);
            activations.add(layer.getActivation().f(output));
        }
    }

    public void update_weights() {

    }

    public void backPropagation(Matrix input, Matrix y_true) {
        forwardPropagation(input);
        Matrix y_pred = activations.get(activations.size() - 1);
        double lossValue = this.loss.f(y_true, y_pred);
        if (this.verbose) {
            System.out.println("Loss: " + lossValue);
        }
        Matrix grad = loss.df(y_true, y_pred);
        for (int i = model.getLayers().size() - 1; i >= 1; i--) {
            Layer layer = model.getLayers().get(i);
            Activation activation = layer.getActivation();
            Matrix activation_grad = grad.elementWiseProduct(activation.df(activations.get(i - 1)));
            gradients.add(activation_grad);
            grad = layer.getWeights().dot(activation_grad);
        }
    }

    public void step(Matrix input, Matrix y_true) {
        backPropagation(input, y_true);
        for (int i = 1; i < model.getLayers().size(); i++) {
            Layer layer = model.getLayers().get(i);
            Matrix grad = gradients.get(gradients.size() - i);
            Matrix weights = layer.getWeights();
            Matrix new_weights = weights
                    .substract(grad.dot(outputs.get(i).transposed()).scale(learning_rate).transposed());
            layer.setWeights(new_weights);
            Matrix biases = layer.getBiases();
            Matrix new_biases = biases.substract(grad.scale(learning_rate));
            layer.setBiases(new_biases);
        }
    }

    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new DenseLayer(2, 6, Activation.ReLU));
        model.add(new DenseLayer(6, 2, Activation.ReLU));
        Matrix input = new Matrix(new double[][] { { 1d }, { 1d } });
        SGD sgd = new SGD(model, 0.01, Loss.MSE, true);
        for (int i = 0; i < 10000; i++) {
            sgd.step(input, new Matrix(new double[][] { { 3.14159265 }, { 0.19012000 } }));
        }
        System.out.println(model.forward(input));
    }
}
