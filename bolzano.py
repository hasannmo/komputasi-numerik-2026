import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def f(x):
    return x**3 - 100

    # return math.cos(x) - 3*x                   # cos x = 3x
    # return math.log(x) - 1 - 1/(x**2)          # ln x = 1 + 1/x^2
    # return math.exp(x * math.log(x)) - 10      # x^x = 10
    # return (1 - 0.6*x) / x                     # (1-0.6x)/x = 0
    # return math.exp(x) - 2*x - 21              # e^x = 2x+21      

def bolzano(x1, x2, n_iter):
    
    fx1 = f(x1)
    fx2 = f(x2)

    if fx1 * fx2 > 0:
        raise ValueError(
            f"\n[ERROR] f(x1) * f(x2) harus <= 0\n"
            f"  f({x1}) = {fx1:.6f}\n"
            f"  f({x2}) = {fx2:.6f}\n"
        )

    rows = []
    for i in range(1, n_iter + 1):
        fx1 = f(x1)
        fx2 = f(x2)
        x3  = (x1 + x2) / 2
        fx3 = f(x3)

        rows.append({
            "Iteration": i,
            "x1":        x1,
            "x2":        x2,
            "x3":        x3,
            "f(x1)":     fx1,
            "f(x2)":     fx2,
            "f(x3)":     fx3,
        })

        if fx1 * fx3 < 0:
            x2 = x3
        else:
            x1 = x3

        if fx3 == 0.0:
            break

    root = rows[-1]["x3"]
    return rows, root

def print_table(rows, decimals=4):
    fmt = f".{decimals}f"
    headers = ["Iteration", "x1", "x2", "x3", "f(x1)", "f(x2)", "f(x3)"]
    table = []
    for r in rows:
        table.append([
            r["Iteration"],
            format(r["x1"],    fmt),
            format(r["x2"],    fmt),
            format(r["x3"],    fmt),
            format(r["f(x1)"], fmt),
            format(r["f(x2)"], fmt),
            format(r["f(x3)"], fmt),
        ])
    print(tabulate(table, headers=headers, tablefmt="outline"))

def plot_function(x1_orig, x2_orig, root, rows):
    
    margin  = abs(x2_orig - x1_orig) * 0.6
    x_lo    = min(x1_orig, x2_orig) - margin
    x_hi    = max(x1_orig, x2_orig) + margin

    xs = np.linspace(x_lo, x_hi, 800)

    ys = []
    for xv in xs:
        try:
            ys.append(f(xv))
        except Exception:
            ys.append(float("nan"))
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#ffffff")

    ax.plot(xs, ys, color="#2563eb", linewidth=2, label="f(x)", zorder=3)

    ax.axhline(0, color="#374151", linewidth=0.8, linestyle="--", zorder=2)

    for xv, label, color in [(x1_orig, "x₁ (start)", "#16a34a"),
                              (x2_orig, "x₂ (start)", "#dc2626")]:
        try:
            ax.axvline(xv, color=color, linewidth=1, linestyle=":", alpha=0.7)
            ax.scatter([xv], [f(xv)], color=color, s=60, zorder=5)
            ax.annotate(label, (xv, f(xv)),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=color)
        except Exception:
            pass

    cmap   = plt.cm.get_cmap("autumn", len(rows))
    for idx, r in enumerate(rows):
        x3v = r["x3"]
        try:
            ax.scatter([x3v], [f(x3v)], color=cmap(idx), s=40,
                       zorder=6, alpha=0.85)
            ax.annotate(f"i={r['Iteration']}", (x3v, f(x3v)),
                        textcoords="offset points", xytext=(4, -14),
                        fontsize=7, color=cmap(idx))
        except Exception:
            pass

    try:
        ax.scatter([root], [0], color="#7c3aed", s=100, zorder=7,
                   label=f"Root ≈ {root:.6f}", marker="*")
        ax.axvline(root, color="#7c3aed", linewidth=1, linestyle="--", alpha=0.5)
    except Exception:
        pass

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.set_title("Bolzano (Bisection) Method — Root Finding", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 55)
    print("   BOLZANO (BISECTION) ROOT FINDER")
    print("=" * 55)

    try:
        x1     = float(input("Enter x1 : "))
        x2     = float(input("Enter x2 : "))
        n_iter = int(input("Number of iterations : "))
    except ValueError:
        print("[ERROR] Invalid input. Enter numeric values.")
        return

    if n_iter < 1:
        print("[ERROR] Number of iterations must be at least 1.")
        return

    print()

    try:
        rows, root = bolzano(x1, x2, n_iter)
    except ValueError as e:
        print(e)
        return

    print_table(rows, decimals=4)
    print()

    print(f"x = {root:.6f}")

    plot_function(x1, x2, root, rows)


if __name__ == "__main__":
    main()
