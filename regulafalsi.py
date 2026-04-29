import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def f(x):
    return x**3 - 100


def regula_falsi(x1, x2, n_iter):
    if f(x1) * f(x2) > 0:
        raise ValueError("f(x1) and f(x2) must have opposite signs.")

    rows = []
    for i in range(1, n_iter + 1):
        fx1, fx2 = f(x1), f(x2)
        x3 = x2 - fx2 * (x1 - x2) / (fx1 - fx2)
        fx3 = f(x3)
        rows.append([i, x1, x2, x3, fx1, fx2, fx3])

        if fx1 * fx3 < 0:
            x2 = x3
        else:
            x1 = x3

        if fx3 == 0:
            break

    return rows


def print_table(rows):
    headers = ["Iter", "x1", "x2", "x3", "f(x1)", "f(x2)", "f(x3)"]
    fmt_rows = [[r[0]] + [f"{v:.4f}" for v in r[1:]] for r in rows]
    print(tabulate(fmt_rows, headers=headers, tablefmt="outline"))


def plot(x1_orig, x2_orig, rows):
    all_x = [v for r in rows for v in [r[1], r[2], r[3]]]
    margin = max(abs(x2_orig - x1_orig) * 0.15, 0.5)
    xs = np.linspace(min(all_x) - margin, max(all_x) + margin, 800)
    ys = np.vectorize(f)(xs)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(rows)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, color="#2563eb", lw=2, label="f(x)", zorder=3)
    ax.axhline(0, color="#374151", lw=0.8, ls="--", zorder=2)

    for r, c in zip(rows, colors):
        i, x1l, x2l, x3l, fx1l, fx2l, fx3l = r
        slope = (fx2l - fx1l) / (x2l - x1l) if x2l != x1l else 0
        sx = np.array([x1l - margin * 0.3, x2l + margin * 0.3])
        ax.plot(sx, fx1l + slope * (sx - x1l),
                color=c, lw=1, ls="--", alpha=0.7)
        ax.scatter([x3l], [0], color=c, s=100, marker="*", zorder=7,
                   label=f"Iter {i}: x3={x3l:.4f}")

    ax.set(xlabel="x", ylabel="f(x)", title="Regula Falsi Semua Iterasi")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    print("REGULA FALSI ROOT FINDER")
    x1 = float(input("Enter x1: "))
    x2 = float(input("Enter x2: "))
    n = int(input("Iterations: "))

    rows = regula_falsi(x1, x2, n)
    print_table(rows)
    print(f"\nRoot ≈ {rows[-1][3]:.6f}")
    plot(x1, x2, rows)


if __name__ == "__main__":
    main()
