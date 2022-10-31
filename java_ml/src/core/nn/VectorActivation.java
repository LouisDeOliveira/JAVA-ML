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

            return getJacobian(input).dot(input);
        }

        private double J(int i, int j, Matrix S) {
            if (i == j) {
                return S.getValue(i, 0) * (1 - S.getValue(j, 0));
            } else {
                return -S.getValue(i, 0) * S.getValue(j, 0);
            }
        }

        private Matrix getJacobian(Matrix input) {
            Matrix S = f(input);
            Matrix res = new Matrix(input.getRows(), input.getRows());
            for (int i = 0; i < input.getRows(); i++) {
                for (int j = 0; j < input.getCols(); j++) {
                    res.setValue(i, j, J(i, j, S));
                }
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
