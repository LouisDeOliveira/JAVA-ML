package core.nn;

import java.io.Serializable;

import core.math.Function;
import core.math.linalg.Matrix;

public abstract class VectorActivation implements Activation, Function<Matrix>, Serializable {
    public static final VectorActivation Softmax = new VectorActivation() {
        public Matrix f(Matrix input) {
            Matrix res = new Matrix(input.getRows(), input.getCols());
            // apparently this provides more stability and does not change the result
            input = input.substract(new Matrix(input.getRows(), input.getCols()).scale(input.max()));
            Matrix expInput = input.map(Function.Exp);

            for (int i = 0; i < input.getRows(); i++) {
                res.setValue(i, 0, expInput.getValue(i, 0) / expInput.sum());
            }

            return res;
        }

        public Matrix df(Matrix input) {
            Matrix res = new Matrix(input.getRows(), input.getCols());
            Matrix expInput = input.map(Function.Exp);
            double sum = expInput.sum();

            for (int i = 0; i < input.getRows(); i++) {
                res.setValue(i, 0, expInput.getValue(i, 0) * (sum - expInput.getValue(i, 0)) / Math.pow(sum, 2));
            }

            return res;
        }
    };

    public static void main(String[] args) {
        Matrix input = new Matrix(new double[][] { { 3 }, { 1 }, { 0.2 } });
        System.out.println(Softmax.f(input));
        System.out.println(Softmax.df(input));
    }

}
