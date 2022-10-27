package core.nn.layers;

import core.math.linalg.Matrix;

/**
 * Dense layer of a neural network.
 */
public abstract class Layer {
    public abstract Matrix forward(Matrix input);
}
