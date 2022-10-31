package core.nn;

import java.io.Serializable;

import core.math.Function;
import core.math.linalg.Matrix;

public abstract class RealActivation implements Activation, Function<Double>, Serializable {

    public static final RealActivation Linear = new RealActivation() {
        @Override
        public Double f(Double input) {
            return input;
        }

        @Override
        public Double df(Double input) {
            return 1d;
        }
    };

    public static final RealActivation Sigmoid = new RealActivation() {
        @Override
        public Double f(Double x) {
            return 1 / (1 + Math.exp(-x));
        }

        @Override
        public Double df(Double x) {
            return f(x) * (1 - f(x));
        }

        @Override
        public String toString() {
            return "Sigmoid";
        }
    };

    public static final RealActivation TanH = new RealActivation() {
        @Override
        public Double f(Double x) {
            return Math.tanh(x);
        }

        @Override
        public Double df(Double x) {
            return 1 - Math.pow(f(x), 2);
        }

        @Override
        public String toString() {
            return "TanH";
        }
    };

    public static final RealActivation ReLU = new RealActivation() {
        @Override
        public Double f(Double x) {
            return Math.max(0, x);
        }

        @Override
        public Double df(Double x) {
            return x > 0d ? 1d : 0d;
        }

        @Override
        public String toString() {
            return "ReLU";
        }
    };

    public static final RealActivation LeakyReLU = new RealActivation() {
        @Override
        public Double f(Double x) {
            return x > 0 ? x : 0.01 * x;
        }

        @Override
        public Double df(Double x) {
            return x > 0 ? 1 : 0.01;
        }

        @Override
        public String toString() {
            return "LeakyReLU: 0.01";
        }
    };

    public static final RealActivation LeakyReLU(Double alpha) {

        RealActivation LeakyReLU = new RealActivation() {
            @Override
            public Double f(Double x) {
                return x > 0 ? x : alpha * x;
            }

            @Override
            public Double df(Double x) {
                return x > 0 ? 1 : alpha;
            }

            @Override
            public String toString() {
                return "LeakyReLU: " + alpha;
            }
        };

        return LeakyReLU;
    }

    public static final RealActivation SELU = new RealActivation() {
        @Override
        public Double f(Double x) {
            return x > 0 ? 1.0507 * x : 1.0507 * 1.67326 * Math.exp(x) - 1.0507 * 1.67326;
        }

        @Override
        public Double df(Double x) {
            return x > 0 ? 1.0507 : 1.0507 * 1.67326 * Math.exp(x);
        }

        @Override
        public String toString() {
            return "SELU";
        }
    };

    /**
     * Applies the function to each element of the matrix.
     * 
     * @param x the matrix to apply the function to
     * @return the resulting matrix
     */
    @Override
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
    @Override
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
