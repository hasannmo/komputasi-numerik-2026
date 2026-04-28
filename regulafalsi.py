import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def f(x):
    return x**3 - 100

    # return math.cos(x) - 3*x                  # cos x = 3x
    # return math.log(x) - 1 - 1/(x**2)         # ln x = 1 + 1/x^2
    # return math.exp(x * math.log(x)) - 10      # x^x = 10
    # return (1 - 0.6*x) / x                     # (1-0.6x)/x = 0
    # return math.exp(x) - 2*x - 21              # e^x = 2x+21

def regula_falsi(x1, x2, n_iter):
    
    fx1 = f(x1)
    fx2 = f(x2)

    if fx1 * fx2 > 0:
        raise ValueError(
            f"\n[ERROR] f(x1) * f(x2) must be <= 0 to guarantee a root in the bracket.\n"
            f"  f({x1}) = {fx1:.6f}\n"
            f"  f({x2}) = {fx2:.6f}\n"
            f"  f(x1) * f(x2) = {fx1 * fx2:.6f}  (positive — no sign change detected)\n"
            f"Please choose a different bracket."
        )

    rows = []
    for i in range(1, n_iter + 1):
        fx1 = f(x1)
        fx2 = f(x2)

        x3  = x2 - fx2 * (x1 - x2) / (fx1 - fx2)
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
    fmt     = f".{decimals}f"
    headers = ["Iteration", "x1", "x2", "x3", "f(x1)", "f(x2)", "f(x3)"]
    table   = []
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

def plot_last_iteration(x1_orig, x2_orig, rows):

    last   = rows[-1]
    x1_l   = last["x1"]
    x2_l   = last["x2"]
    x3_l   = last["x3"]
    fx1_l  = last["f(x1)"]
    fx2_l  = last["f(x2)"]
    fx3_l  = last["f(x3)"]
    it_num = last["Iteration"]

    
    margin = max(abs(x2_l - x1_l) * 0.8, abs(x2_orig - x1_orig) * 0.15)
    x_lo   = min(x1_l, x2_l) - margin
    x_hi   = max(x1_l, x2_l) + margin

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

    secant_xs = np.array([x1_l - margin * 0.3, x2_l + margin * 0.3])
    slope     = (fx2_l - fx1_l) / (x2_l - x1_l) if (x2_l - x1_l) != 0 else 0
    secant_ys = fx1_l + slope * (secant_xs - x1_l)
    ax.plot(secant_xs, secant_ys, color="#f59e0b", linewidth=1.6,
            linestyle="--", label="Secant line (interpolation)", zorder=4)


    ax.scatter([x1_l], [fx1_l], color="#16a34a", s=80, zorder=6,
               label=f"x₁ = {x1_l:.4f},  f(x₁) = {fx1_l:.4f}")
    ax.plot([x1_l, x1_l], [0, fx1_l], color="#16a34a",
            linewidth=1, linestyle=":", zorder=5)
    ax.annotate(f"x₁={x1_l:.4f}", (x1_l, fx1_l),
                textcoords="offset points", xytext=(-38, 8),
                fontsize=8, color="#16a34a")

    ax.scatter([x2_l], [fx2_l], color="#dc2626", s=80, zorder=6,
               label=f"x₂ = {x2_l:.4f},  f(x₂) = {fx2_l:.4f}")
    ax.plot([x2_l, x2_l], [0, fx2_l], color="#dc2626",
            linewidth=1, linestyle=":", zorder=5)
    ax.annotate(f"x₂={x2_l:.4f}", (x2_l, fx2_l),
                textcoords="offset points", xytext=(6, 8),
                fontsize=8, color="#dc2626")

    ax.scatter([x3_l], [0], color="#7c3aed", s=120, zorder=7,
               marker="*", label=f"x₃ = {x3_l:.4f}  (root estimate, iter {it_num})")
    ax.plot([x3_l, x3_l], [0, fx3_l], color="#7c3aed",
            linewidth=1, linestyle=":", alpha=0.6, zorder=5)
    ax.annotate(f"x₃={x3_l:.4f}\nf(x₃)={fx3_l:.4f}",
                (x3_l, 0),
                textcoords="offset points", xytext=(8, -28),
                fontsize=8, color="#7c3aed")

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.set_title(
        f"Regula Falsi — Last Iteration (Iteration {it_num})",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 55)
    print("   REGULA FALSI (FALSE POSITION) ROOT FINDER")
    print("=" * 55)

    try:
        x1     = float(input("Enter x1 : "))
        x2     = float(input("Enter x2 : "))
        n_iter = int(input("Number of iterations : "))
    except ValueError:
        print("[ERROR] Invalid input. Please enter numeric values.")
        return

    if n_iter < 1:
        print("[ERROR] Number of iterations must be at least 1.")
        return

    print()

    
    try:
        rows, root = regula_falsi(x1, x2, n_iter)
    except ValueError as e:
        print(e)
        return

    
    print_table(rows, decimals=4)
    print()


    print(f"x = {root:.6f}")


    plot_last_iteration(x1, x2, rows)


if __name__ == "__main__":
    main()
