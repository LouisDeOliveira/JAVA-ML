package core.nn;

import java.io.Serializable;

import core.math.Function;
import core.math.linalg.Matrix;

public abstract class Activation implements Function, Serializable {

    public static final Activation Sigmoid = new Activation() {
        @Override
        public double f(double x) {
            return 1 / (1 + Math.exp(-x));
        }

        @Override
        public double df(double x) {
            return f(x) * (1 - f(x));
        }
    };

    public static final Activation TanH = new Activation() {
        @Override
        public double f(double x) {
            return Math.tanh(x);
        }

        @Override
        public double df(double x) {
            return 1 - Math.pow(f(x), 2);
        }
    };

    public static final Activation ReLU = new Activation() {
        @Override
        public double f(double x) {
            return Math.max(0, x);
        }

        @Override
        public double df(double x) {
            return x > 0 ? 1 : 0;
        }
    };

    public static final Activation LeakyReLU = new Activation() {
        @Override
        public double f(double x) {
            return x > 0 ? x : 0.01 * x;
        }

        @Override
        public double df(double x) {
            return x > 0 ? 1 : 0.01;
        }
    };

    public static final Activation LeakyReLU(double alpha) {

        Activation LeakyReLU = new Activation() {
            @Override
            public double f(double x) {
                return x > 0 ? x : alpha * x;
            }

            @Override
            public double df(double x) {
                return x > 0 ? 1 : alpha;
            }
        };

        return LeakyReLU;
    }

    public static final Activation SELU = new Activation() {
        @Override
        public double f(double x) {
            return x > 0 ? 1.0507 * x : 1.0507 * 1.67326 * Math.exp(x) - 1.0507 * 1.67326;
        }

        @Override
        public double df(double x) {
            return x > 0 ? 1.0507 : 1.0507 * 1.67326 * Math.exp(x);
        }
    };

    /**
     * Applies the function to each element of the matrix.
     * 
     * @param x the matrix to apply the function to
     * @return the resulting matrix
     */
    public Matrix f(Matrix x) {
        Matrix y = new Matrix(x.getRows(), x.getCols());
        for (int i = 0; i < x.getRows(); i++) {
            for (int j = 0; j < x.getCols(); j++) {
                y.setValue(i, j, f(x.getValue(i, j)));
            }
        }
        return y;
    }

    /**
     * Applies the derivative of the function to each element of the matrix.
     * 
     * @param x the matrix to apply the derivative of the function to
     * @return the resulting matrix
     */
    public Matrix df(Matrix x) {
        Matrix y = new Matrix(x.getRows(), x.getCols());
        for (int i = 0; i < x.getRows(); i++) {
            for (int j = 0; j < x.getCols(); j++) {
                y.setValue(i, j, df(x.getValue(i, j)));
            }
        }
        return y;
    }

}
