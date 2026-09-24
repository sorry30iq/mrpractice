/**
 * Задача 3. Является ли целое число a числовым палиндромом.
 * Пример: 121 - палиндром, 123 - нет.
 */
public class PalindromeNumber {

    /**
     * Математический вариант: разворачиваем число делением/остатком, без строк.
     * Сложность: O(log10(a)) - количество итераций равно числу цифр в a,
     * а число цифр n-значного числа ~ log10(a). Память O(1).
     */
    public static boolean isPalindrome(int a) {
        if (a < 0) return false; // отрицательные числа не считаем палиндромами
        if (a != 0 && a % 10 == 0) return false; // числа вида 10, 100... не палиндромы (кроме 0)

        int original = a;
        long reversed = 0; // long на случай переполнения int при развороте
        while (a > 0) {
            reversed = reversed * 10 + a % 10;
            a /= 10;
        }
        return original == reversed;
    }

    /**
     * Альтернатива через строку, для сравнения (тоже валидна).
     * Сложность аналогичная: O(log10(a)) на создание строки и её разворот.
     */
    public static boolean isPalindromeString(int a) {
        if (a < 0) return false;
        String s = String.valueOf(a);
        String reversed = new StringBuilder(s).reverse().toString();
        return s.equals(reversed);
    }

    public static void main(String[] args) {
        int[] tests = {121, 123, 1221, -121, 0, 7, 1000021};
        for (int t : tests) {
            System.out.printf("%d -> математически: %b, через строку: %b%n",
                    t, isPalindrome(t), isPalindromeString(t));
        }
    }
}
