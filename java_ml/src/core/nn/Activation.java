package core.nn;

import core.math.Function;

public abstract class Activation extends Function {

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

}
