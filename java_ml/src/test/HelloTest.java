package test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class HelloTest {

    @Test
    public void testHello() {
        System.out.println("Hello World!");
        assertEquals("Hello World!", "Hello World!");
    }
}
