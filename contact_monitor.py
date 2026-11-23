import numpy as np


class ContactForceController:
    """PI force regulation along the first detected contact normal."""

    def __init__(
        self,
        detect_force_n,
        target_force_n,
        kp,
        ki,
        deadband_n,
        max_velocity_m_s,
        dt,
    ):
        self.detect_force_n = float(detect_force_n)
        self.target_force_n = float(target_force_n)
        self.kp = float(kp)
        self.ki = float(ki)
        self.deadband_n = float(deadband_n)
        self.max_velocity_m_s = float(max_velocity_m_s)
        self.dt = float(dt)
        if not np.isfinite(self.detect_force_n) or self.detect_force_n <= 0.0:
            raise ValueError("detection force must be positive")
        if not np.isfinite(self.target_force_n) or self.target_force_n <= 0.0:
            raise ValueError("target force must be positive")
        if not np.isfinite(self.kp) or self.kp < 0.0:
            raise ValueError("kp must be non-negative")
        if not np.isfinite(self.ki) or self.ki < 0.0:
            raise ValueError("ki must be non-negative")
        if not np.isfinite(self.deadband_n) or self.deadband_n < 0.0:
            raise ValueError("force deadband must be non-negative")
        if not np.isfinite(self.max_velocity_m_s) or self.max_velocity_m_s <= 0.0:
            raise ValueError("maximum contact velocity must be positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.active = False
        self.detected_force_n = 0.0
        self.normal = np.zeros(3)
        self.integral = 0.0

    def update(self, wrench):
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must contain six finite values")
        force = wrench[:3]
        force_norm = float(np.linalg.norm(force))
        if not self.active and force_norm >= self.detect_force_n:
            self.active = True
            self.detected_force_n = force_norm
            self.normal = force / force_norm
            self.integral = 0.0
        if not self.active:
            return False, np.zeros(6), 0.0

        normal_force = float(np.dot(force, self.normal))
        force_error = normal_force - self.target_force_n
        if abs(force_error) <= self.deadband_n:
            self.integral *= 0.9
            normal_velocity = 0.0
        else:
            self.integral += force_error * self.dt
            if self.ki > 0.0:
                integral_limit = self.max_velocity_m_s / self.ki
                self.integral = float(
                    np.clip(
                        self.integral,
                        -integral_limit,
                        integral_limit,
                    )
                )
            normal_velocity = self.kp * force_error + self.ki * self.integral
            normal_velocity = float(
                np.clip(
                    normal_velocity,
                    -self.max_velocity_m_s,
                    self.max_velocity_m_s,
                )
            )

        velocity = np.zeros(6)
        velocity[:3] = normal_velocity * self.normal
        return True, velocity, normal_force
