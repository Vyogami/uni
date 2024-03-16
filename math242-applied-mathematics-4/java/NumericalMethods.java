package appliedmathematics;

public class NumericalMethods {


    public static double bisection(double a, double b, double TOLERANCE, Function function) {
        double c = (a+b)/2;

        if(function.fx(a)*function.fx(b) >= 0){
            throw new IllegalArgumentException("Invalid range [" + a + ", " + b + "]");
        }

        if (a > b) {
            double temp = a;
            a = b;
            b = temp;
        }

        while(b-a >= TOLERANCE){
            c = (a+b)/2;

            if(function.fx(c) == 0.0){
                break;
            }
            if(function.fx(a)*function.fx(c) < 0){
                b = c; 
            }
            else{
                a = c;
            }
        }
        return c;
    }

    public static double bisection(double a, double b, double TOLERANCE, Function function, boolean print) {
        double c = (a+b)/2;
        int count = 1;

        if(function.fx(a)*function.fx(b) >= 0){
            throw new IllegalArgumentException("Invalid range [" + a + ", " + b + "]");
        }

        if (a > b) {
            double temp = a;
            a = b;
            b = temp;
        }

        while(b-a >= TOLERANCE){
            c = (a+b)/2;
            if(print){
                System.out.println("x" + count + ": " + c); 
            }

            if(function.fx(c) == 0.0){
                break;
            }
            if(function.fx(a)*function.fx(c) < 0){
                b = c; 
            }
            else{
                a = c;
            }

            count++;
        }
        return c;
    }

}
