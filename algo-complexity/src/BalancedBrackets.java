import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;

/**
 * Задача 4. Строка s содержит символы, включая скобки '(', ')', '{', '}', '[', ']'.
 * Проверить, что каждой открывающей скобке соответствует ровно одна закрывающая того же типа.
 */
public class BalancedBrackets {

    private static final Map<Character, Character> PAIRS = Map.of(
            ')', '(',
            '}', '{',
            ']', '['
    );

    /**
     * Классическое решение через стек.
     * Сложность: O(n) по времени (один проход по строке),
     * O(n) по памяти в худшем случае (строка из одних открывающих скобок).
     */
    public static boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();

        for (char c : s.toCharArray()) {
            if (PAIRS.containsValue(c)) {
                // открывающая скобка - кладём в стек
                stack.push(c);
            } else if (PAIRS.containsKey(c)) {
                // закрывающая скобка - должна совпасть с вершиной стека
                if (stack.isEmpty() || stack.pop() != PAIRS.get(c)) {
                    return false;
                }
            }
            // остальные символы игнорируем - в строке могут быть и не-скобки
        }

        return stack.isEmpty();
    }

    public static void main(String[] args) {
        String[] tests = {
                "()",
                "()[]{}",
                "(]",
                "([)]",
                "{[]}",
                "a(b[c]d){e}",
                "(((",
        };

        for (String t : tests) {
            System.out.printf("\"%s\" -> %b%n", t, isValid(t));
        }
    }
}
