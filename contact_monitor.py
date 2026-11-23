import numpy as np


class ContactForceController:
    """Hybrid free-space and contact-force controller."""

    def __init__(
        self,
        detect_force_n,
        target_force_n,
        kp,
        ki,
        deadband_n,
        max_velocity_m_s,
        dt,
        confirm_time_s=0.10,
        max_direction_angle_deg=20.0,
        release_after_s=0.0,
        release_time_s=0.50,
        release_complete_force_n=0.50,
    ):
        self.detect_force_n = float(detect_force_n)
        self.target_force_n = float(target_force_n)
        self.kp = float(kp)
        self.ki = float(ki)
        self.deadband_n = float(deadband_n)
        self.max_velocity_m_s = float(max_velocity_m_s)
        self.dt = float(dt)
        self.confirm_time_s = float(confirm_time_s)
        self.max_direction_angle_deg = float(max_direction_angle_deg)
        self.release_after_s = float(release_after_s)
        self.release_time_s = float(release_time_s)
        self.release_complete_force_n = float(release_complete_force_n)
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
        if not np.isfinite(self.confirm_time_s) or self.confirm_time_s <= 0.0:
            raise ValueError("contact confirmation time must be positive")
        if not 0.0 < self.max_direction_angle_deg < 90.0:
            raise ValueError("direction angle must be in (0, 90) degrees")
        if not np.isfinite(self.release_after_s) or self.release_after_s < 0.0:
            raise ValueError("release delay must be non-negative")
        if not np.isfinite(self.release_time_s) or self.release_time_s <= 0.0:
            raise ValueError("release time must be positive")
        if (
            not np.isfinite(self.release_complete_force_n)
            or self.release_complete_force_n < 0.0
        ):
            raise ValueError("release force threshold must be non-negative")

        self.state = "free"
        self.detected_force_n = 0.0
        self.normal = np.zeros(3)
        self.integral = 0.0
        self.confirmation_time_s = 0.0
        self.contact_time_s = 0.0
        self.release_elapsed_s = 0.0
        self.release_target_force_n = self.target_force_n

    def _tangential_velocity(self, admittance_velocity):
        velocity = np.asarray(admittance_velocity, dtype=float).copy()
        normal_speed = float(np.dot(velocity[:3], self.normal))
        velocity[:3] -= normal_speed * self.normal
        return velocity

    def _force_velocity(self, normal_force, target_force_n):
        force_error = normal_force - target_force_n
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
        return velocity

    def update(self, wrench, admittance_velocity):
        wrench = np.asarray(wrench, dtype=float)
        admittance_velocity = np.asarray(admittance_velocity, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must contain six finite values")
        if admittance_velocity.shape != (6,) or not np.all(
            np.isfinite(admittance_velocity)
        ):
            raise ValueError("admittance velocity must contain six finite values")

        force = wrench[:3]
        force_norm = float(np.linalg.norm(force))

        if self.state == "free":
            if force_norm < self.detect_force_n:
                return "free", admittance_velocity.copy(), 0.0
            self.state = "confirming"
            self.detected_force_n = force_norm
            self.normal = force / force_norm
            self.confirmation_time_s = 0.0
            self.integral = 0.0

        if self.state == "confirming":
            if force_norm < self.detect_force_n * 0.8:
                self.state = "free"
                self.confirmation_time_s = 0.0
                return "free", admittance_velocity.copy(), 0.0
            direction = force / force_norm
            alignment = float(np.dot(direction, self.normal))
            minimum_alignment = np.cos(
                np.deg2rad(self.max_direction_angle_deg)
            )
            if alignment < minimum_alignment:
                self.normal = direction
                self.confirmation_time_s = 0.0
            else:
                self.normal = 0.8 * self.normal + 0.2 * direction
                self.normal /= np.linalg.norm(self.normal)
                self.confirmation_time_s += self.dt
            normal_force = float(np.dot(force, self.normal))
            velocity = self._tangential_velocity(admittance_velocity)
            if self.confirmation_time_s >= self.confirm_time_s:
                self.state = "contact"
                self.contact_time_s = 0.0
            return self.state, velocity, normal_force

        normal_force = float(np.dot(force, self.normal))
        if self.state == "contact":
            self.contact_time_s += self.dt
            if (
                self.release_after_s > 0.0
                and self.contact_time_s >= self.release_after_s
            ):
                self.state = "release"
                self.release_elapsed_s = 0.0
                self.release_target_force_n = self.target_force_n
            else:
                velocity = self._tangential_velocity(admittance_velocity)
                velocity += self._force_velocity(normal_force, self.target_force_n)
                return self.state, velocity, normal_force

        if self.state == "release":
            self.release_elapsed_s += self.dt
            ratio = min(1.0, self.release_elapsed_s / self.release_time_s)
            self.release_target_force_n = self.target_force_n * (1.0 - ratio)
            velocity = self._force_velocity(
                normal_force,
                self.release_target_force_n,
            )
            if (
                ratio >= 1.0
                and normal_force <= self.release_complete_force_n
            ):
                self.state = "released"
                return self.state, np.zeros(6), normal_force
            return self.state, velocity, normal_force

        return "released", np.zeros(6), normal_force
