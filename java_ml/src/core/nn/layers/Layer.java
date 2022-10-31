package core.nn.layers;

import core.math.linalg.Matrix;

/**
 * Dense layer of a neural network.
 */
public abstract class Layer {

    public abstract Matrix forward(Matrix input);

    public abstract Matrix backward(Matrix input, Matrix gradOutput);

    public abstract void applyGradient(Matrix input, Matrix gradOutput);

    public abstract boolean isTrainable();

    public abstract boolean isTraining();

    public abstract void setTraining(boolean training);

}
