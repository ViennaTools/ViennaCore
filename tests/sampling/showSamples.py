# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
samples = np.genfromtxt("../../build/tests/sampling/samples_1D.txt")
samples_alias = np.genfromtxt("../../build/tests/sampling/samples_1D_alias.txt")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
nbins = 40

# create histogram of samples
x = np.linspace(-3, 3, 1000)
expected = 1 / np.sqrt(np.pi) * np.exp(-(x**2))

ax1.hist(samples, bins=nbins, density=True, alpha=0.6, color="g", label="Samples")
ax1.plot(x, expected, "r", label="Expected")
ax1.set(xlabel="x", ylabel="Probability", title="Histogram of samples")
ax1.legend(loc="upper right")

ax2.hist(
    samples_alias, bins=nbins, density=True, alpha=0.6, color="g", label="Alias Samples"
)
ax2.plot(x, expected, "r", label="Expected")
ax2.set(xlabel="x", ylabel="Probability", title="Histogram of samples (alias)")
ax2.legend(loc="upper right")

plt.show()

# %%
samples_2D = np.genfromtxt("../../build/tests/sampling/samples_2D.txt", delimiter=",")

# %%
# create histogram of samples

h, xedges, yedges, _ = plt.hist2d(
    samples_2D[:, 0], samples_2D[:, 1], bins=30, density=True, alpha=0.6, color="g"
)

plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Histogram of samples")

plt.show()

# %%
# generate custom distribution

x = np.linspace(5, 95, 1000)
y = 1 / np.sin(x * np.pi / 100)

norm = (
    (np.log(np.tan(x[-1] * np.pi / 100 / 2)) - np.log(np.tan(x[0] * np.pi / 100 / 2)))
    * 100
    / np.pi
)

np.savetxt(
    "../../build/tests/sampling/custom_distribution.txt", np.column_stack((x, y))
)

# %%
samples_custom = np.genfromtxt("../../build/tests/sampling/samples_custom.txt")

plt.hist(samples_custom, bins=50, density=True, alpha=0.6, color="g", label="Samples")
plt.plot(x, y / norm, "r", label="Expected")
plt.legend(loc="upper right")
plt.show()

# %%
