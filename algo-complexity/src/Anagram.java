import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Задача 1. Проверка, является ли строка a анаграммой строки b.
 *
 * Анаграмма - слово, образованное перестановкой букв другого слова.
 * Пример: "пила" и "липа".
 */
public class Anagram {

    /**
     * Вариант 1: через сортировку символов.
     * Сложность: O(n log n) по времени (сортировка), O(n) по памяти (копии массивов).
     */
    public static boolean isAnagramSorting(String a, String b) {
        if (a.length() != b.length()) return false;
        char[] ca = a.toCharArray();
        char[] cb = b.toCharArray();
        Arrays.sort(ca);
        Arrays.sort(cb);
        return Arrays.equals(ca, cb);
    }

    /**
     * Вариант 2 (оптимальный): подсчёт частот символов через HashMap.
     * Сложность: O(n) по времени (два линейных прохода), O(k) по памяти,
     * где k - размер алфавита (константа, не зависит от n).
     */
    public static boolean isAnagramFrequency(String a, String b) {
        if (a.length() != b.length()) return false;

        Map<Character, Integer> count = new HashMap<>();
        for (char c : a.toCharArray()) {
            count.merge(c, 1, Integer::sum);
        }
        for (char c : b.toCharArray()) {
            Integer current = count.get(c);
            if (current == null || current == 0) return false;
            count.put(c, current - 1);
        }
        return true;
    }

    public static void main(String[] args) {
        String[][] tests = {
                {"пила", "липа"},
                {"пост", "стоп"},
                {"кот", "ток"},
                {"кот", "пёс"}
        };

        for (String[] t : tests) {
            boolean bySort = isAnagramSorting(t[0], t[1]);
            boolean byFreq = isAnagramFrequency(t[0], t[1]);
            System.out.printf("\"%s\" / \"%s\" -> sorting=%b, frequency=%b%n",
                    t[0], t[1], bySort, byFreq);
        }
    }
}
