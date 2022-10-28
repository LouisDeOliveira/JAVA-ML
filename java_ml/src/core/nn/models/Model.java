package core.nn.models;

import core.math.linalg.Matrix;

public abstract class Model {

    public abstract Matrix forward(Matrix input);
}
