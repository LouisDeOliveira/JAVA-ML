package core.nn;

import core.math.linalg.*;

public abstract class Loss {

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
}