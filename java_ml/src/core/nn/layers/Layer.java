package core.nn.layers;

import core.math.linalg.Matrix;
import core.nn.Activation;

/**
 * Dense layer of a neural network.
 */
public abstract class Layer {

    public abstract Matrix forward(Matrix input);

    public abstract Matrix backward(Matrix input, Matrix gradOutput);

}
