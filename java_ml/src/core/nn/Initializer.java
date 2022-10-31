package core.nn;

import java.io.Serializable;
import java.util.Random;

public abstract class Initializer implements Serializable {
    public static final Initializer NormalInitializer(double mean, double std) {
        return new Initializer() {

            Random random = new Random();

            @Override
            public double[][] initialize(int[] shape) {
                double[][] weights = new double[shape[0]][shape[1]];

                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        weights[i][j] = random.nextGaussian() * std + mean;
                    }
                }

                return weights;
            }
        };
    }

    public static final Initializer UniformInitializer(double min, double max) {
        return new Initializer() {

            Random random = new Random();

            @Override
            public double[][] initialize(int[] shape) {
                double[][] weights = new double[shape[0]][shape[1]];

                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        weights[i][j] = random.nextDouble() * (max - min) + min;
                    }
                }

                return weights;
            }
        };
    }

    public static final Initializer BinaryInitializer(float p) {
        return new Initializer() {

            Random random = new Random();

            @Override
            public double[][] initialize(int[] shape) {
                double[][] weights = new double[shape[0]][shape[1]];

                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        boolean value = random.nextFloat() > p;
                        weights[i][j] = (double) (value ? 1 : 0);
                    }
                }

                return weights;
            }
        };
    }

    public abstract double[][] initialize(int[] shape);
}
