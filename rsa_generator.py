__author__ = "sion"

import sys
from random import randint

def get_rsa(d: int):
    """
    RSA PK generation program with the modulus n and exponent e^2.

    Some theory from A3 specs...
    Modular n = p * q
        p, q = smallest two prime integers (2^d − 1)
        d ≥ d
            d > 2,
            d ∈ Z+,
            d ≤ 2000,

    Exponent e = random([3, λ − 1])
        gcd(e, λ) = 1, i.e. e and λ are relatively prime,
        λ = (p−1)(q−1) / gcd(p−1,q−1)

    :param d: The starting value for prime number search.
    """
    # Get p, q and n = p*q
    p, q = get_prime_nums(d)
    n = p * q

    # Get gcd_num = gcd(p-1, q-1)
    gcd_num = get_gcd(p - 1, q - 1)

    # Get λ = (p−1)(q−1) / gcd_num
    lamb = ((p - 1) * (q - 1)) // gcd_num

    # Get e = random([3, λ − 1])
    e = generate_e(lamb)

    # Write results to the output files
    output(n, e, p, q)

def generate_e(lamb):
    """
    Generates a random e such that gcd(e, λ) = 1, from the range [3, λ-1].
    :param lamb: λ, to check if e and λ are coprime.
    :return e: The public key exponent.
    """
    while True:
        e = randint(3, lamb - 1)
        if get_gcd(e, lamb) == 1:  # The GCD of e and lambda must be 1 (coprime)
            return e

def get_prime_nums(d):
    """
    Returns the two smallest prime numbers of (2^d - 1).
    :param d: Starting value for prime number search.
    :return prime_nums: The list of two smallest and distinct prime numbers.
    """
    prime_nums = []

    # Assign k based on the size of d
    # 'k' determines the accuracy of miller-rabin test
    # Assumption from the specs: d ≤ 2000

    if d < 500:
        k = 20  # Fewer tests for a smaller `d`
    elif d < 1500:
        k = 30  # Moderate number of tests
    else:
        k = 40  # More tests for a large `d` approaching 2000

    while len(prime_nums) < 2:  # get two prime numbers
        prime_num = (2 ** d) - 1
        if test_miller_rabin(prime_num, k):
            prime_nums.append(prime_num)  # append the prime numbers
        d += 1

    return prime_nums

def test_miller_rabin(n, k):
    """
    Miller-Rabin randomised primality test.

    :param n: The number to test for primality.
    :param k: The number of rounds of miller rabin testing. (Accuracy control)
    :return: Boolean to indicate if the number is probably prime.
    """
    # Base cases for smaller n
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n < 2:
        return False

    # n-1 as 2^s * d
    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # Test k times with random bases
    for i in range(k):
        a = randint(2, n - 2)  # random base
        x = modular_exp(a, d, n)  # modular exponent

        if x == 1 or x == n - 1:
            continue

        for j in range(s - 1):
            x = modular_exp(x, 2, n)
            if x == n - 1:
                break
        else:
            # n is composite
            return False

    # n is probably prime
    return True

def modular_exp(base: int, exp: int, mod: int):
    """
    Modular exponentiation with repeated squaring.

    :param base: The base.
    :param exp: The exponent.
    :param mod: The modulus.
    :return result: base^exp % mod
    """
    result = 1
    base = base % mod

    while exp > 0:
        # If exp is odd, multiply base with result
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp // 2  # exp must be even now
        base = (base * base) % mod

    return result

def get_gcd(a: int, b: int) -> int:
    """
    Euclid algorithm to compute the greatest common divisor of any two numbers.

    :param a: The first number.
    :param b: The second number.
    :return int: The GCD of a and b.
    """
    # Iterative determination of GCD
    while b != 0:
        a, b = b, a % b
    return a

def output(n, e, p, q):
    """
    Generates the output file.

    1. output_q1_public.txt
        # modulus (n)
        1073602561
        # exponent (e)
        3187811
    2. output_q1_private.txt
        # p
        8191
        # q
        131071

    :param n: The modulus. (PK)
    :param e: The public exponent. (PK)
    :param p: The first prime number. (SK)
    :param q: The second prime number. (SK)
    """
    with open("output_q1_public.txt", "w+") as f:
        f.write("# modulus (n)\n")
        f.write("%d\n" % n)
        f.write("# exponent (e)\n")
        f.write("%d\n" % e)

    with open("output_q1_private.txt", "w+") as f:
        f.write("# p\n")
        f.write("%d\n" % p)
        f.write("# q\n")
        f.write("%d\n" % q)


if __name__ == '__main__':
    # python q1.py <d>
    d_val = int(sys.argv[1])
    get_rsa(d_val)
