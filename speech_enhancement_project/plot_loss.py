import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Run 1 (3 epochs)
tr1 = [0.0329, 0.0278, 0.0264]
va1 = [0.0165, 0.0148, 0.0148]

# Run 2 (25 epochs, cosine schedule)
tr2 = [0.0259,0.0252,0.0248,0.0243,0.0241,0.0239,0.0236,0.0234,0.0233,0.0230,
       0.0229,0.0227,0.0226,0.0225,0.0223,0.0222,0.0221,0.0220,0.0219,0.0219,
       0.0218,0.0218,0.0218,0.0218,0.0217]
va2 = [0.0150,0.0147,0.0144,0.0144,0.0146,0.0155,0.0144,0.0143,0.0142,0.0143,
       0.0143,0.0140,0.0138,0.0139,0.0140,0.0139,0.0139,0.0138,0.0140,0.0138,
       0.0138,0.0138,0.0139,0.0138,0.0138]

train = tr1 + tr2
val = va1 + va2
epochs = list(range(1, len(train) + 1))

plt.figure(figsize=(8, 4.5))
plt.plot(epochs, train, label="Train loss")
plt.plot(epochs, val, label="Validation loss")
plt.axvline(3.5, color="gray", linestyle="--", linewidth=1, label="Start of cosine schedule")
plt.xlabel("Epoch")
plt.ylabel("L1 loss (log1p magnitude)")
plt.title("U-Net speech enhancement: training curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

out = os.path.join("results", "plots")
os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "loss_curve.png"), dpi=200)
print("saved", os.path.join(out, "loss_curve.png"), "| epochs:", len(epochs))