import numpy as np
import matplotlib.pyplot as plt

# Parameters
vi = 20.0       # Initial speed (m/s)
vf = 0.0        # Final speed (m/s)
distance = 50.0 # Desired stopping distance (m)
time_const = 5.0  # Time to stop with constant deceleration
num_points = 200

# 1) Constant Deceleration Profile
# --------------------------------
# We'll assume it takes exactly 5 seconds to go from 20 m/s to 0.
# Deceleration a_const = (vf - vi)/time_const = -4 m/s^2
t_const = np.linspace(0, time_const, num_points)
speed_const = vi + (vf - vi)*(t_const / time_const)
# Distance under constant deceleration: s = vi*t + 0.5*a*t^2
a_const = (vf - vi)/time_const
distance_const = vi*t_const + 0.5*a_const*(t_const**2)

# 2) Jerk-Limited (S-Curve) Profile
# ---------------------------------
# We'll do a simplified approach:
# Phase 1: Ramp from a=0 to a=a_max with jerk j
# Phase 2: Maintain a_max
# Phase 3: Ramp from a=a_max to 0 with -j
# We'll pick j so that the total distance is ~50 m. This is a toy example.
a_max = -4.0   # target deceleration
j = -2.0       # jerk (m/s^3)

# Time to ramp to a_max: t1 = a_max / j  (since a_max is negative, j is negative, be mindful of sign)
t1 = abs(a_max / j)
# We'll guess that we hold a_max for some time t2, then ramp back to 0 in time t3 = t1
# Let's see if we can solve it so total distance is 50 m

# For simplicity, we can do a numeric approach to simulate step by step and see if we hit 50 m around 5 seconds
dt = 0.01
time_scurve = []
speed_scurve = []
distance_scurve = []
acc_scurve = []

v = vi
s = 0
t = 0
phase = 1
while True:
    if phase == 1:  # ramp from 0 to a_max
        a = j*(t) if (t <= t1) else a_max
        if t > t1:
            phase = 2
    elif phase == 2: # hold a_max until speed < 0 or next phase
        a = a_max
        # If speed is about to go below 0 next step, time to ramp up
        if v + a*dt <= 0:
            phase = 3
    else:  # phase 3: ramp from a_max to 0
        # We'll ramp acceleration from a_max to 0 in time t1
        # Let tau = t - (t1 + ???). We'll track it differently to keep it simple
        # We'll just keep ramping up acceleration linearly over t1
        a = a_max - j*(t - (t1 + 1.0))  # This "1.0" is an approximation to start the ramp
        if a > 0:
            a = 0

    # update speed and distance
    v_new = v + a*dt
    # stop if speed < 0
    if v_new < 0:
        v_new = 0
    s_new = s + v*dt + 0.5*a*(dt**2)

    time_scurve.append(t)
    speed_scurve.append(v)
    distance_scurve.append(s)
    acc_scurve.append(a)

    v = v_new
    s = s_new
    t += dt

    # break if we've gone 6+ seconds or traveled >= 50 m
    if s >= distance or t > 6.0:
        break

# Convert to numpy arrays for plotting
time_scurve = np.array(time_scurve)
speed_scurve = np.array(speed_scurve)
distance_scurve = np.array(distance_scurve)

# PLOTTING
# 1) Speed vs. Time
plt.figure(figsize=(8,5))
plt.plot(t_const, speed_const, label='Constant Deceleration', linestyle='-', color='blue')
plt.plot(time_scurve, speed_scurve, label='Jerk-Limited S-Curve', linestyle='--', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed vs Time Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Distance vs. Time
plt.figure(figsize=(8,5))
plt.plot(t_const, distance_const, label='Constant Deceleration', linestyle='-', color='blue')
plt.plot(time_scurve, distance_scurve, label='Jerk-Limited S-Curve', linestyle='--', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance vs Time Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
