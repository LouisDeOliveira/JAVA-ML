package core.optim;

import java.util.ArrayList;

import core.math.linalg.Matrix;
import core.nn.RealActivation;
import core.nn.VectorActivation;
import core.nn.Initializer;
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
    private ArrayList<Matrix> gradients;

    public SGD(Model model, double learning_rate, Loss loss) {
        this.model = model;
        this.learning_rate = learning_rate;
        this.loss = loss;
        this.verbose = false;
        this.outputs = new ArrayList<Matrix>();
        this.gradients = new ArrayList<Matrix>();
    }

    public SGD(Model model, double learning_rate, Loss loss, boolean verbose) {
        this.model = model;
        this.learning_rate = learning_rate;
        this.loss = loss;
        this.verbose = verbose;
        this.outputs = new ArrayList<Matrix>();
        this.gradients = new ArrayList<Matrix>();
    }

    public void forwardPropagation(Matrix input) {
        outputs = new ArrayList<Matrix>();

        outputs.add(input);

        for (int i = 0; i < model.getLayers().size(); i++) {
            Layer layer = model.getLayers().get(i);
            Matrix output = layer.forward(outputs.get(i));
            outputs.add(output);
        }
    }

    public void backPropagation(Matrix input, Matrix y_true) {
        gradients = new ArrayList<Matrix>();
        forwardPropagation(input);
        Matrix y_pred = outputs.get(outputs.size() - 1);
        double lossValue = this.loss.f(y_true, y_pred);
        if (this.verbose) {
            System.out.println("Loss: " + lossValue);
        }
        Matrix grad = loss.df(y_true, y_pred);
        gradients.add(grad);
        for (int i = model.getLayers().size() - 1; i > 0; i--) {
            Layer layer = model.getLayers().get(i);
            grad = layer.backward(outputs.get(i), grad);
            if (layer.isTrainable()) {
                gradients.add(grad);
            }
        }
    }

    public void step(Matrix input, Matrix y_true) {
        backPropagation(input, y_true);
        int current_layer = 1;
        for (int i = 0; i < model.getLayers().size() - 1; i++) {
            Layer layer = model.getLayers().get(i);
            if (layer.isTrainable()) {
                Matrix grad = gradients.get(gradients.size() - current_layer);
                Matrix output = outputs.get(i);
                layer.applyGradient(output, grad.scale(learning_rate));
                current_layer++;
            }

        }
    }

    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new DenseLayer(2, 16));
        model.add(new ActivationLayer(RealActivation.ReLU));
        model.add(new DenseLayer(16, 6));
        model.add(new ActivationLayer(RealActivation.ReLU));
        model.add(new DenseLayer(6, 3));
        model.add(new ActivationLayer(VectorActivation.Softmax));
        Matrix true_m = new Matrix(new double[][] { { 0, 0, 1 } }).transposed();
        Matrix input = new Matrix(new double[][] { { 1d }, { 1d } });
        SGD sgd = new SGD(model, 0.01, Loss.MSE, true);
        for (int i = 0; i < 1000; i++) {
            sgd.step(input, true_m);
        }
        System.out.println(model.forward(input));
        System.out.println(true_m);
    }
}
