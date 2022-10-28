package core.nn.layers;

import core.math.linalg.Matrix;
import core.nn.Activation;

/**
 * Dense layer of a neural network.
 */
public abstract class Layer {

    public abstract Matrix forward(Matrix input);

    public abstract void backward(Matrix biasGradient, Matrix weightGradient);

    public abstract Matrix getActivationValues();

    public abstract Activation getActivation();

    public abstract Matrix getWeights();

    public abstract Matrix getBiases();

    public abstract void setWeights(Matrix weights);

    public abstract void setBiases(Matrix biases);

}
