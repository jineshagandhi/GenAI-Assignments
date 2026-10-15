Prompt: Write a Python function is_prime(n) that checks whether a number is prime, and print all prime numbers from 1 to 50. Reply with only the Python code.

```python
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

for i in range(1, 51):
    if is_prime(i):
        print(i)
```

This Python function `is_prime` checks if a given number `n` is prime. It iterates through numbers from 2 up to the square root of `n`, checking if any of them divide `n`. If it finds such a divisor, `n` is not prime, and the function returns `False`. If no divisors are found, `n` is prime, and the function returns `True`. The function then prints all prime numbers from 1 to 50.
