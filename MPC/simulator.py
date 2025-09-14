from stepper_functions import*
# --- Kezdő állapot ---
xk = x0_val.copy()
X_traj = [xk]

for k in range(N-1):
    xk = rk4_step(xk, 0)
    X_traj.append(xk.full().flatten())

X_traj = np.array(X_traj)  
plt.figure()
    
plt.plot(t,X_traj[:,2])
plt.plot(t,yoyo_height(X_traj[:,0],X_traj[:,2]))
plt.grid()
plt.show()
