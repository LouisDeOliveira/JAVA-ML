package core.math;

/**
 * Interface for a real-valued function. R->R
 *
 */
public interface Function<T> {
    public T f(T x);

    public T df(T x);

    public static final Function<Double> Exp = new Function<Double>() {
        @Override
        public Double f(Double input) {
            return Math.exp(input);
        }

        @Override
        public Double df(Double input) {
            return Math.exp(input);
        }
    };

}
