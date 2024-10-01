Operator Conventions in Kotlin.

Kotlin allows you to provide implementations for a predefined set of operators for your types. These operator functions are defined by applying the `operator` modifier to a function definition. This enables intuitive usage of operators like `+`, `-`, `[]`, and others for custom types.

Arithmetic Operators

You can overload arithmetic operators such as `+`, `-`, `*`, `/`, and `%` by defining the following functions:
- `plus()`
- `minus()`
- `times()`
- `div()`
- `rem()`

Example:
```kotlin
data class Vector(val x: Int, val y: Int) {
    operator fun plus(other: Vector) = Vector(x + other.x, y + other.y)
}

val v1 = Vector(1, 2)
val v2 = Vector(3, 4)
val sum = v1 + v2 // Calls v1.plus(v2)
```

Unary Operators

Unary operators such as `+a`, `-a`, `!a`, `++a`, and `--a` can be overloaded with:
- `unaryPlus()`
- `unaryMinus()`
- `not()`
- `inc()` for `++`
- `dec()` for `--`

Example:
```kotlin
data class Point(val x: Int, val y: Int) {
    operator fun unaryMinus() = Point(-x, -y)
}

val p = Point(1, 2)
val negativeP = -p // Calls negativeP.unaryMinus(-p)
```

Comparison Operators

Comparison operators are mapped to functions:
- `==` and `!=` use `equals()` (already provided by `Any`).
- `<`, `>`, `<=`, `>=` need to use `compareTo()`.

Example:
```kotlin
data class Person(val age: Int): Comparable<Person> {
    override fun compareTo(other: Person) = this.age - other.age
}

val p1 = Person(25)
val p2 = Person(30)
println(p1 < p2) // Calls p1.compareTo(p2)
```

Assignment Operators

Assignment operators such as `+=`, `-=`, `*=`, `/=`, and `%=` can be overloaded by defining:
- `plusAssign()`
- `minusAssign()`
- `timesAssign()`
- `divAssign()`

Example:
```kotlin
class Counter(var value: Int) {
    operator fun plusAssign(increment: Int) {
        value += increment
    }
}

val counter = Counter(10)
counter += 5 // Calls counter.plusAssign(5)
```

Index Access

To enable indexing with `[]`, you can define:
- `get()` for reading values
- `set()` for assigning values

Example:
```kotlin
class Matrix(val rows: Int, val cols: Int) {
    private val data = Array(rows) { IntArray(cols) }

    operator fun get(i: Int, j: Int): Int = data[i][j]
    operator fun set(i: Int, j: Int, value: Int) {
        data[i][j] = value
    }
}

val matrix = Matrix(3, 3)
matrix[0, 0] = 10 // Calls matrix.set(0, 0, 10)
println(matrix[0, 0]) // Calls matrix.get(0, 0)
```

Invoke Operator

You can overload the function call operator `()` by implementing the `invoke()` function. This allows instances of a class to be called like functions.

Example:
```kotlin
class Greeter(val greeting: String) {
    operator fun invoke(name: String) {
        println("$greeting, $name!")
    }
}

val greeter = Greeter("Hello")
greeter("Kotlin") // Calls greeter.invoke("Kotlin")
```

Destructuring Declarations

To enable destructuring declarations for a class, implement methods called `componentN()`, where `N` is a number starting from 1.

Example:
```kotlin
data class Point(val x: Int, val y: Int)

val (x, y) = Point(10, 20) // Calls Point(10, 20).component1() and Point(10, 20).component2()
```

You can use Extension Functions for operator overload. 
Yes, you can overload operators for basic types like String using Kotlin's extension functions. However, there's an important restriction: you cannot override or modify the existing behavior of operators for Kotlin's built-in types (such as String, Int, etc.).

Example: Overloading `+` for `String` and `Int` (custom behavior)

Let's say we want to overload the `+` operator so that when we add a `String` and an `Int`, it repeats the string that many times.

```kotlin
// Extension function to overload the + operator for String and Int
operator fun String.plus(times: Int): String {
    return this.repeat(times)
}

fun main() {
    val text = "Hello"

    val repeatedText = text + 3  // text + 3 is translated into text.plus(3)
    println(repeatedText)        // Output: HelloHelloHello
}
```



In Kotlin, you can combine multiple operator conventions in one expression. Below is an example demonstrating the simultaneous use of several operators such as indexing (`[]`), assignment (`+=`), and increment (`++`) in a single expression.

Example: Using multiple operators in one expression

We will create a `Matrix` class that supports:
1. **Indexing (`[]`)**: To access elements.
2. **The `+=` operator**: To add values to elements.
3. **The post-increment operator (`++`)**: To increment the value of an element.

```kotlin
class Matrix(val rows: Int, val cols: Int) {
    private val data = Array(rows) { IntArray(cols) }

    // Overload the get operator to access elements of the matrix
    operator fun get(row: Int, col: Int): Int {
        return data[row][col]
    }

    // Overload the set operator to assign values to the matrix
    operator fun set(row: Int, col: Int, value: Int) {
        data[row][col] = value
    }

    // Overload the += operator for matrix elements
    operator fun plusAssign(value: Int) {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j] += value
            }
        }
    }

    // Overload the ++ operator for the matrix
    operator fun inc(): Matrix {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j]++
            }
        }
        return this
    }

    // Print the matrix for convenience
    fun printMatrix() {
        for (row in data) {
            println(row.joinToString(" "))
        }
        println()
    }
}

fun main() {
    // Create a 3x3 matrix
    val matrix = Matrix(3, 3)

    // Set the value at index (0, 0) to 5
    matrix[0, 0] = 5

    // Increment the value at (0, 0) and add it to (0, 1)
    var incrementValue = matrix[0, 0]++
    matrix[0, 1] += incrementValue

    // Print the result
    matrix.printMatrix() // Output:
                         // 6 5 0
                         // 0 0 0
                         // 0 0 0
}
```

A More Complex Example: Combining Multiple Operators

In this more advanced example, we will create a `Matrix2D` class that supports operations with `Vector` objects, allowing for multiple operator conventions like indexing and assignment to be used in the same expression.

```kotlin
class Vector(val size: Int) {
    private val data = IntArray(size)

    // Overload get operator for the vector
    operator fun get(index: Int): Int = data[index]

    // Overload set operator for the vector
    operator fun set(index: Int, value: Int) {
        data[index] = value
    }

    // Overload the += operator for vector elements
    operator fun plusAssign(value: Int) {
        for (i in 0 until size) {
            data[i] += value
        }
    }

    // Overload the ++ operator for vector
    operator fun inc(): Vector {
        for (i in 0 until size) {
            data[i]++
        }
        return this
    }

    // Print vector for convenience
    fun printVector() {
        println(data.joinToString(" "))
    }
}

class Matrix2D(val rows: Int, val cols: Int) {
    private val data = Array(rows) { Vector(cols) }

    // Overload get operator to return a row (vector) from the matrix
    operator fun get(row: Int): Vector = data[row]

    // Overload set operator to assign a vector to a row
    operator fun set(row: Int, vector: Vector) {
        data[row] = vector
    }

    // Print matrix for convenience
    fun printMatrix() {
        for (row in data) {
            row.printVector()
        }
        println()
    }
}

fun main() {
    // Create a 3x3 matrix
    val matrix = Matrix2D(3, 3)

    // Set the first element of the first row
    matrix[0][0] = 5

    // Increment the first element of the first row and add it to the entire first row
    matrix[0] += matrix[0][0]++

    // Print the result
    matrix.printMatrix()  // Output:
                          // 6 6 6
                          // 0 0 0
                          // 0 0 0
}
```
