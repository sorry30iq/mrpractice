import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Задача 2. В несортированном массиве найти индексы двух чисел,
 * сумма которых равна заданному k. Если пары нет - вернуть null.
 */
public class TwoSum {

    /**
     * Наивный вариант: перебор всех пар.
     * Сложность: O(n^2) по времени, O(1) по дополнительной памяти.
     */
    public static int[] twoSumBruteForce(int[] nums, int k) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == k) {
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }

    /**
     * Оптимальный вариант: одна хэш-таблица "значение -> индекс".
     * Сложность: O(n) по времени (один проход, O(1) поиск/вставка в среднем),
     * O(n) по памяти под саму хэш-таблицу.
     */
    public static int[] twoSumHashMap(int[] nums, int k) {
        Map<Integer, Integer> seen = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int need = k - nums[i];
            if (seen.containsKey(need)) {
                return new int[]{seen.get(need), i};
            }
            seen.put(nums[i], i);
        }
        return null;
    }

    public static void main(String[] args) {
        int[] nums = {2, 7, 11, 15, -3, 8};
        int k = 9;

        System.out.println("Массив: " + Arrays.toString(nums) + ", k = " + k);
        System.out.println("Brute force: " + Arrays.toString(twoSumBruteForce(nums, k)));
        System.out.println("HashMap:     " + Arrays.toString(twoSumHashMap(nums, k)));

        int[] noPair = {1, 2, 3};
        System.out.println("Без решения (k=100): " + Arrays.toString(twoSumHashMap(noPair, 100)));
    }
}
