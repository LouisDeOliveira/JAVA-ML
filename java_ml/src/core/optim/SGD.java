package core.optim;

import core.math.linalg.Matrix;
import core.nn.Activation;
import core.nn.Loss;
import core.nn.models.Model;
import core.nn.models.Sequential;
import core.nn.layers.*;

public class SGD {
    double learning_rate;
    Model model;
    Loss loss;

    public SGD(Model model, double learning_rate, Loss loss) {
        this.model = model;
        this.learning_rate = learning_rate;
        this.loss = loss;
    }

    public void step(Matrix input, Matrix y_true, Matrix y_pred) {
        double lossValue = this.loss.f(y_true, y_pred);
        System.out.println("loss = " + lossValue);
        Matrix grad = loss.df(y_true, y_pred);
        for (int i = model.getLayers().size() - 1; i >= 0; i--) {
            Layer layer = model.getLayers().get(i);
            Layer previousLayer = i > 0 ? model.getLayers().get(i - 1) : null;
            Matrix activation = layer.getActivationValues();
            grad = grad.elementWiseProduct(layer.getActivation().df(activation));

            Matrix previousActivationValueT = previousLayer != null
                    ? previousLayer.getActivation().f(previousLayer.getActivationValues()).transposed()
                    : input.transposed();

            Matrix biasGradient = grad;
            Matrix weightGradient = grad
                    .dot(previousActivationValueT);

            layer.setWeights(layer.getWeights().substract(weightGradient.scale(learning_rate).transposed()));
            layer.setBiases(layer.getBiases().substract(biasGradient.scale(learning_rate)));

            grad = layer.getWeights().dot(grad);

        }
    }

    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new DenseLayer(2, 16, Activation.ReLU));
        model.add(new DenseLayer(16, 8, Activation.ReLU));
        model.add(new DenseLayer(8, 16, Activation.ReLU));
        model.add(new DenseLayer(16, 2, Activation.TanH));
        Matrix input = new Matrix(new double[][] { { 1d }, { -1d } });
        Matrix output = model.forward(input);
        System.out.println("output = \n" + output);
        SGD sgd = new SGD(model, 0.001, Loss.MSE);
        for (int i = 0; i < 10000; i++) {
            sgd.step(input, new Matrix(new double[][] { { 1d }, { -1d } }), output);
            output = model.forward(input);
            System.out.println(output);
        }

    }
}
