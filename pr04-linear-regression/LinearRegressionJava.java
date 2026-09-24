import java.util.Random;

/**
 * Практическая работа 4 - Java-версия.
 * В методичке (п.3.2) явно указано: "Реализуйте модель линейной регрессии
 * на выбранном ЯП (python, JS, etc.)" - то есть язык не принципиален,
 * поэтому дублируем реализацию на Java без внешних библиотек.
 *
 * Модель: y = b + a*x, обучение - градиентный спуск по MSE.
 */
public class LinearRegressionJava {

    public static void main(String[] args) {
        Random rnd = new Random(42);
        int n = 100;

        double aTrue = 2.0, bTrue = 1.0;
        double[] x = new double[n];
        double[] y = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextGaussian();
            double eps = 0.1 * rnd.nextGaussian();
            y[i] = bTrue + aTrue * x[i] + eps;
        }

        // train/test split 70/30
        int trainSize = 70;
        double[] xTrain = new double[trainSize];
        double[] yTrain = new double[trainSize];
        double[] xTest = new double[n - trainSize];
        double[] yTest = new double[n - trainSize];
        System.arraycopy(x, 0, xTrain, 0, trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(x, trainSize, xTest, 0, n - trainSize);
        System.arraycopy(y, trainSize, yTest, 0, n - trainSize);

        double a = rnd.nextGaussian();
        double b = rnd.nextDouble();
        double lr = 0.01;
        int epochs = 200;

        for (int ep = 0; ep < epochs; ep++) {
            double sumErr = 0, sumErrX = 0, sumSqErr = 0;
            for (int i = 0; i < trainSize; i++) {
                double pred = b + a * xTrain[i];
                double error = pred - yTrain[i];
                sumErr += error;
                sumErrX += error * xTrain[i];
                sumSqErr += error * error;
            }
            double loss = sumSqErr / trainSize;
            double bGrad = 2 * sumErr / trainSize;
            double aGrad = 2 * sumErrX / trainSize;

            a -= lr * aGrad;
            b -= lr * bGrad;

            if (ep % 40 == 0 || ep == epochs - 1) {
                System.out.printf("epoch %3d: loss=%.6f a=%.4f b=%.4f%n", ep, loss, a, b);
            }
        }

        System.out.printf("%nИстинные параметры: a=%.1f, b=%.1f%n", aTrue, bTrue);
        System.out.printf("Обученные параметры: a=%.4f, b=%.4f%n", a, b);

        // оценка на тесте (MSE)
        double testSumSqErr = 0;
        for (int i = 0; i < xTest.length; i++) {
            double pred = b + a * xTest[i];
            double error = pred - yTest[i];
            testSumSqErr += error * error;
        }
        double testMse = testSumSqErr / xTest.length;
        System.out.printf("MSE на тестовой выборке: %.5f%n", testMse);
    }
}
