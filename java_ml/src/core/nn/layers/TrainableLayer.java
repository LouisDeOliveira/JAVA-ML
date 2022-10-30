package core.nn.layers;

import core.math.linalg.Matrix;

public interface TrainableLayer {
    public void applyGradient(Matrix input, Matrix gradient);
}
