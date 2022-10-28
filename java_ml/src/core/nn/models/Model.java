package core.nn.models;

import core.math.linalg.Matrix;
import core.nn.layers.Layer;
import java.util.*;

public abstract class Model {

    public abstract Matrix forward(Matrix input);

    public abstract ArrayList<Layer> getLayers();

    public abstract void setLayer(int index, Layer layer);
}
