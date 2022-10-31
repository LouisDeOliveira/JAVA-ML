package core.nn;

import java.io.Serializable;

import core.math.Function;
import core.math.linalg.*;

public abstract class Loss implements Serializable {

    public static final Loss MSE = new Loss() {
        @Override
        public double f(Matrix y_true, Matrix y_pred) {
            int N_samples = y_true.getCols();
            int N_dims = y_true.getRows();
            double loss = 0.0;

            for (int i = 0; i < N_samples; i++) {
                Matrix ye = y_true.getCol(i).substract(y_pred.getCol(i));
                loss += ye.transposed().dot(ye).unitMatrixAsDouble() / N_dims;
            }

            return loss / N_samples;
        }

        @Override
        public Matrix df(Matrix y, Matrix y_pred) {
            Matrix mean_grad = Matrix.zeros(y.getRows(), 1);
            int N_samples = y.getCols();
            int N_dims = y.getRows();
            for (int i = 0; i < N_samples; i++) {
                mean_grad = mean_grad.add(y_pred.getCol(i).substract(y.getCol(i))).scale(2.0 / N_dims);
            }
            return mean_grad.scale(1.0 / N_samples);
        }
    };

    public static final Loss BinaryCrossentropy = new Loss() {
        @Override
        public double f(Matrix y_true, Matrix y_pred) {
            y_pred = y_pred.clip(1e-9, 1 - 1e-9);
            Matrix logy_hat = y_pred.map(Function.Log);
            // System.out.println("logy_hat: \n" + logy_hat);
            Matrix one = Matrix.ones(y_pred.getSize()).scale(0.9999999);
            Matrix log1_minus_y_hat = one.substract(y_pred).map(Function.Log);
            // System.out.println("logy-yhat: \n" + log1_minus_y_hat);
            Matrix ylog_y_hat = y_true.elementWiseProduct(logy_hat);
            // System.out.println("ylog_yhat: \n" + ylog_y_hat);
            Matrix one_minus_y = one.substract(y_true);
            // System.out.println("one_minus_y: \n" + one_minus_y);
            return ylog_y_hat.add(one_minus_y.elementWiseProduct(log1_minus_y_hat)).scale(-1.d).mean();
        }

        @Override
        public Matrix df(Matrix y, Matrix y_pred) {
            return y_pred.substract(y).scale(1.d / y_pred.getCols());
        }
    };

    /**
     * Compute the loss between the predicted and target values.
     * The matrix can be seen as a batch of vectors.
     * 
     * @param y_pred The predicted values.
     * @param y_true The target values.
     * @return The loss.
     */
    public abstract double f(Matrix y_true, Matrix y_pred);

    /**
     * Compute the gradient of the loss with respect to the predicted values.
     * The matrix can be seen as a batch of vectors.
     * 
     * @param y_pred The predicted values.
     * @param y_true The target values.
     * @return The derivative of the loss.
     */
    public abstract Matrix df(Matrix y_true, Matrix y_pred);

    public static void main(String[] args) {
        Matrix y_pred = new Matrix(new double[][] { { 0.0 }, { 0.1 }, { 0.9 } });
        Matrix y_true = new Matrix(new double[][] { { 0. }, { 0. }, { 1. } });
        System.out.println(BinaryCrossentropy.f(y_true, y_pred));
        System.out.println(BinaryCrossentropy.df(y_true, y_pred));
    }
}