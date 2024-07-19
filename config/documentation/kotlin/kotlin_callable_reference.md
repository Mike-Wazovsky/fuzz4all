Reflection is a set of language and library features that allows you to introspect the structure of your program at runtime. 
Functions and properties are first-class citizens in Kotlin, and the ability to introspect them (for example, 
learning the name or the type of a property or function at runtime) is essential when using a functional or reactive style.

Callable references.
References to functions, properties, and constructors can be called or used as instances of function types.

The common supertype for all callable references is KCallable<out R>, where R is the return value type. 
It is the property type for properties, and the constructed type for constructors.

Function references
When you have a named function declared as below, you can call it directly (isOdd(5)):
fun isOdd(x: Int) = x % 2 != 0

Alternatively, you can use the function as a function type value, that is, pass it to another function. To do so, use the :: operator:
val numbers = listOf(1, 2, 3)
println(numbers.filter(::isOdd))

Here ::isOdd is a value of function type (Int) -> Boolean.

Function references belong to one of the KFunction<out R> subtypes, depending on the parameter count. For instance, KFunction3<T1, T2, T3, R>.

:: can be used with overloaded functions when the expected type is known from the context. For example:
fun isOdd(x: Int) = x % 2 != 0
fun isOdd(s: String) = s == "brillig" || s == "slithy" || s == "tove"

val numbers = listOf(1, 2, 3)
println(numbers.filter(::isOdd)) // refers to isOdd(x: Int)

Alternatively, you can provide the necessary context by storing the method reference in a variable with an explicitly specified type:

val predicate: (String) -> Boolean = ::isOdd   // refers to isOdd(x: String)
If you need to use a member of a class or an extension function, it needs to be qualified: String::toCharArray.

Even if you initialize a variable with a reference to an extension function, the inferred function type will have no receiver, but it will have an additional parameter accepting a receiver object. To have a function type with a receiver instead, specify the type explicitly:

val isEmptyStringList: List<String>.() -> Boolean = List<String>::isEmpty
Example: function composition
Consider the following function:

fun <A, B, C> compose(f: (B) -> C, g: (A) -> B): (A) -> C {
    return { x -> f(g(x)) }
}
It returns a composition of two functions passed to it: compose(f, g) = f(g(*)). You can apply this function to callable references:

fun length(s: String) = s.length

val oddLength = compose(::isOdd, ::length)
val strings = listOf("a", "ab", "abc")

println(strings.filter(oddLength))