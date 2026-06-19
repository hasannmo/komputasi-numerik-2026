def f(x):
    #change function
    return 1 / (1 + x ** 2)


def trapezoidal(a, b, n):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def romberg(a, b, itr):
    R = [[0.0] * itr for _ in range(itr)]

    for i in range(itr):
        n = 2 ** i
        R[i][0] = trapezoidal(a, b, n)
        for k in range(1, i + 1):
            R[i][k] = (4 ** k * R[i][k - 1] - R[i - 1][k - 1]) / (4 ** k - 1)

    return R, R[itr - 1][itr - 1]


def print_table(R, itr):
    col_width = 16
    header = "  i |" + "".join(f"R[i,{k}]".rjust(col_width) for k in range(itr))
    print(header)
    print("-" * len(header))

    for i in range(itr):
        row = f"{i:>3} |"
        for k in range(itr):
            if k <= i:
                row += f"{R[i][k]:.10f}".rjust(col_width)
            else:
                row += " " * col_width
        print(row)


def main():
    print("=== Romberg Integration ===\n")

    a = float(input("Enter lower limit a: "))
    b = float(input("Enter upper limit b: "))
    itr = int(input("itr: "))

    if itr < 1:
        print("Number of iterations must be at least 1.")
        return

    R, result = romberg(a, b, itr+1)

    print_table(R, itr+1)

    print(f"\nIntegral ≈ {result:.10f}\n")


if __name__ == "__main__":
    main()
