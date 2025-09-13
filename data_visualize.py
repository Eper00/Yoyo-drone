import pandas as pd
import matplotlib.pyplot as plt

# CSV beolvasása
df = pd.read_csv("data_mujoco_30.csv", header=0)
# mintavételi idő (s)
dt = 0.001
time = df.index * dt   # idővektor

# Theta, Theta_dot, Theta_ddot
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Yoyo Szögdinamika", fontsize=14)

axs[0].plot(time, df["Theta"], color="blue", linewidth=0.9)
axs[0].set_ylabel("Theta [rad]")
axs[0].set_title("Szög (Theta)")
axs[0].grid(True)

axs[1].plot(time, df["Theta_dot"], color="green", linewidth=0.9)    
axs[1].set_ylabel("Theta_dot [rad/s]")
axs[1].set_title("Szögsebesség (Theta_dot)")
axs[1].grid(True)

axs[2].plot(time, df["Theta_ddot"], color="red", linewidth=0.9)
axs[2].set_ylabel("Theta_ddot [rad/s²]")
axs[2].set_xlabel("Idő [s]")
axs[2].set_title("Szöggyorsulás (Theta_ddot)")
axs[2].grid(True)

plt.show()

# z, z_dot, z_ddot
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Yoyo Pozíciódinamika", fontsize=14)

axs[0].plot(time, df["z"], color="blue", linewidth=0.9)
axs[0].set_ylabel("z [m]")
axs[0].set_title("Pozíció (z)")
axs[0].grid(True)

axs[1].plot(time, df["z_dot"], color="green", linewidth=0.9)
axs[1].set_ylabel("z_dot [m/s]")
axs[1].set_title("Sebesség (z_dot)")
axs[1].grid(True)
axs[2].plot(time, df["z_ddot"], color="red", linewidth=0.9)
axs[2].set_ylabel("z_ddot [m/s²]")
axs[2].set_xlabel("Idő [s]")
axs[2].set_title("Gyorsulás (z_ddot)")
axs[2].grid(True)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Drón állapota és az ideális állapotok", fontsize=14)
axs[0].plot(time, df["h"], color="red", linewidth=0.9,label="Elérendő trajektória")
axs[0].plot(time, df["pos"], color="blue", linewidth=0.9,label="Elért trajektória")
axs[0].legend()
axs[0].set_ylabel("[m]")
axs[0].set_title("Pozíció (z)")
axs[0].grid(True)

axs[1].plot(time, df["h_dot"], color="red", linewidth=0.9,label="Elérendő trajektória")
axs[1].plot(time, df["vel"], color="blue", linewidth=0.9,label="Elért trajektória")
axs[1].legend()
axs[1].set_ylabel("[m/s]")
axs[1].set_title("Sebesség")
axs[1].grid(True)
axs[2].plot(time, df["h_ddot"], color="red", linewidth=0.9,label="Elérendő trajektória")
axs[2].plot(time, df["acc"], color="blue", linewidth=0.9,label="Elért trajektória")
axs[2].legend()
axs[2].set_ylabel("[m/s²]")
axs[2].set_xlabel("Idő [s]")
axs[2].set_title("Gyorsulás")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# z, z_dot, z_ddot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Erők", fontsize=14)

axs[0].plot(time, df["tension"], color="blue", linewidth=0.9)
axs[0].set_ylabel("[N]")
axs[0].set_title("Kötél erő")
axs[0].grid(True)

axs[1].plot(time, df["f"], color="green", linewidth=0.9)
axs[1].set_ylabel("[N]")
axs[1].set_title("Alkalmazott properrel erő")
axs[1].grid(True)
plt.tight_layout()
plt.show()