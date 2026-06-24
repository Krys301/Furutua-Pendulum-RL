import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FurutaEnv(gym.Env):
    """
    Furuta (Rotary Inverted) Pendulum Environment
    Based on EE291 Group 4 - Maynooth University
    State: [theta1, theta2, theta1_dot, theta2_dot, i]
    Action: motor voltage V in [-5, 5]
    """

    def __init__(self):
        super().__init__()

        # Physical parameters from EE291 report
        self.g   = 9.81
        self.L1  = 0.334
        self.l1  = 0.180
        self.l2  = 0.178
        self.m1  = 0.360
        self.m2  = 0.090
        self.J1  = 4.29e-2
        self.J2  = 6.67e-3
        self.b1  = 1.73e-4
        self.b2  = 4.84e-4
        self.Lm  = 0.006
        self.Rm  = 9.36
        self.Km  = 0.108

        # Derived inertia terms
        self.J1_hat = self.J1 + self.m1 * self.l1**2
        self.J2_hat = self.J2 + self.m2 * self.l2**2
        self.J0_hat = self.J1_hat + self.m2 * self.L1**2

        # Simulation timestep
        self.dt = 0.01  # 10ms — matches real control loop speeds

        # Action space: motor voltage [-5V, 5V]
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(1,), dtype=np.float32
        )

        # Observation space: [theta1, theta2, theta1_dot, theta2_dot, i]
        high = np.array([np.pi, np.pi, 10.0, 10.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        #start near upright position with small random perturbation
        #theta2 = 0 means upright (deviation from equilibrium)
        self.state = np.array([0.0,self.np_random.uniform(-0.05,0.05),0.0,0.0,0.0],dtype =np.float32)

        return self.state, {}
    def step(self, action):
        V = float(np.clip(action[0], -5.0, 5.0))

        theta1, theta2, theta1_dot, theta2_dot, i = self.state

        # Shorthand
        m2, L1, l2 = self.m2, self.L1, self.l2
        b1, b2     = self.b1, self.b2
        g          = self.g
        J0         = self.J0_hat
        J2         = self.J2_hat
        Km, Rm, Lm = self.Km, self.Rm, self.Lm

        # Torque from motor current
        T1 = Km * i
        T2 = 0.0  # no disturbance torque

        # Denominator (shared by both equations 33 and 34)
        den = (J0 * J2 + J2**2 * np.sin(theta2)**2
               - m2**2 * L1**2 * l2**2 * np.cos(theta2)**2)

        # Equation 33 — theta1 angular acceleration
        num1 = (
            (-J2 * b1) * theta1_dot
            + (m2 * L1 * l2 * np.cos(theta2) * b2) * theta2_dot
            + (-J2 * np.sin(2 * theta2)) * theta1_dot * theta2_dot
            + (-0.5 * J2 * m2 * L1 * l2 * np.cos(theta2) * np.sin(2 * theta2)) * theta1_dot**2
            + (J2 * m2 * L1 * l2 * np.sin(theta2)) * theta2_dot**2
            + J2 * T1
            + (-m2 * L1 * l2 * np.cos(theta2)) * T2
            + 0.5 * m2**2 * l2**2 * L1**2 * np.sin(2 * theta2) * g
        )
        theta1_ddot = num1 / den

        # Equation 34 — theta2 angular acceleration
        num2 = (
            (m2 * L1 * l2 * np.cos(theta2) * b1) * theta1_dot
            + (-b2 * (J0 + J2 * np.sin(theta2)**2)) * theta2_dot
            + (m2 * L1 * l2 * np.cos(theta2) * np.sin(2 * theta2)) * theta1_dot * theta2_dot
            + (-0.5 * np.sin(2 * theta2) * (J0 + J2 * np.sin(theta2)**2)) * theta1_dot**2
            + (-0.5 * m2**2 * L1**2 * l2**2 * np.sin(2 * theta2)) * theta2_dot**2
            + (-m2 * L1 * l2 * np.cos(theta2)) * T1
            + (J0 + J2 * np.sin(theta2)**2) * T2
            + (-m2 * l2 * np.sin(theta2) * (J0 + J2 * np.sin(theta2)**2)) * g
        )
        theta2_ddot = num2 / den

        # Motor current dynamics (Kirchhoff's law)
        i_dot = (V - Rm * i - Km * theta1_dot) / Lm

        # Euler integration — update state
        theta1     += theta1_dot     * self.dt
        theta2     += theta2_dot     * self.dt
        theta1_dot += theta1_ddot   * self.dt
        theta2_dot += theta2_ddot   * self.dt
        i          += i_dot          * self.dt

        self.state = np.array(
            [theta1, theta2, theta1_dot, theta2_dot, i],
            dtype=np.float32
        )

        # Reward — agent gets +1 for every step pendulum stays upright
        # penalise large pendulum angle and large control effort
        reward = float(1.0
            - 5.0 * float(theta2)**2
            - 0.1 * float(V)**2
            - 0.5 * float(theta1_dot)**2
        )

        # Episode ends if pendulum falls too far (>150 degrees)
        terminated = bool(abs(theta2) > np.deg2rad(150))
        truncated  = False
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        return self.state, reward, terminated, truncated, {}
    def render(self):
        pass  # we'll add visualisation later
