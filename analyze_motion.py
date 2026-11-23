"""Summarize admittance CSV logs and analyze force-to-velocity coupling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, coherence, csd, sosfiltfilt, welch


AXES = ("x", "y", "z", "rx", "ry", "rz")


def first_column(rows, names, required=True):
    for name in names:
        if name in rows[0]:
            return name
    if required:
        raise ValueError(f"CSV has none of these columns: {names}")
    return None


def column(rows, names, required=True):
    name = first_column(rows, names, required=required)
    if name is None:
        return None
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def highpass(values, sample_hz, cutoff_hz):
    if len(values) < 32 or cutoff_hz >= sample_hz / 2.0:
        return np.zeros_like(values)
    sos = butter(4, cutoff_hz, btype="highpass", fs=sample_hz, output="sos")
    return sosfiltfilt(sos, values)


def dominant_frequency(values, sample_hz):
    nperseg = min(256, len(values))
    frequencies, power = welch(
        values - values.mean(),
        fs=sample_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    valid = (frequencies >= 0.2) & (frequencies <= min(20.0, sample_hz / 2.0))
    if not np.any(valid):
        return float("nan")
    return float(frequencies[valid][np.argmax(power[valid])])


def force_velocity_response(force, velocity, sample_hz):
    nperseg = min(256, len(force))
    frequencies, sff = welch(
        force - force.mean(),
        fs=sample_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    _, svv = welch(
        velocity - velocity.mean(),
        fs=sample_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    _, svf = csd(
        force - force.mean(),
        velocity - velocity.mean(),
        fs=sample_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        _, coh = coherence(
            force,
            velocity,
            fs=sample_hz,
            nperseg=nperseg,
            noverlap=nperseg // 2,
        )
    valid = (frequencies >= 0.2) & (frequencies <= min(20.0, sample_hz / 2.0))
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")
    gain = np.abs(svf[valid]) / np.maximum(sff[valid], 1e-15)
    frequency = frequencies[valid][np.argmax(coh[valid])]
    index = np.argmin(np.abs(frequencies[valid] - frequency))
    return float(frequency), float(gain[index]), float(coh[valid][index])


def analyze_file(path, highpass_hz, contact_threshold, start=None, end=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: CSV is empty")
    time = column(rows, ("time_s",))
    mask = np.ones(len(rows), dtype=bool)
    if start is not None:
        mask &= time >= start
    if end is not None:
        mask &= time <= end
    selected = [row for row, keep in zip(rows, mask) if keep]
    if len(selected) < 8:
        raise ValueError(f"{path}: not enough rows in analysis window")
    time = column(selected, ("time_s",))
    dt = np.diff(time)
    dt = dt[dt > 0.0]
    sample_hz = 1.0 / float(np.median(dt))

    axes_result = {}
    for axis in AXES:
        force = column(
            selected,
            (
                f"F_ext_{axis}",
                f"F_filtered_{axis}",
                f"F_raw_{axis}",
            ),
        )
        raw_velocity = column(
            selected,
            (f"v_actual_raw_{axis}", f"v_actual_{axis}"),
        )
        control_velocity = column(
            selected,
            (f"v_control_{axis}", f"v_actual_{axis}"),
        )
        admittance_acceleration = column(
            selected,
            (f"a_admittance_raw_{axis}",),
            required=False,
        )
        actual_acceleration = column(
            selected,
            (f"a_actual_{axis}",),
            required=False,
        )
        jerk = column(
            selected,
            (f"jerk_actual_{axis}",),
            required=False,
        )
        target = column(
            selected,
            (f"x_target_{axis}", f"x_cmd_{axis}"),
            required=False,
        )
        actual = column(selected, (f"x_actual_{axis}",))
        servo = column(
            selected,
            (f"x_servo_{axis}", f"x_cmd_{axis}"),
            required=False,
        )
        tracking = column(
            selected,
            (
                f"x_servo_minus_actual_{axis}",
                f"x_cmd_minus_actual_{axis}",
            ),
            required=False,
        )

        def rms(values):
            return None if values is None else float(np.sqrt(np.mean(values**2)))

        def hp_rms(values):
            if values is None:
                return None
            return float(np.sqrt(np.mean(highpass(values, sample_hz, highpass_hz) ** 2)))

        response_frequency, response_gain, response_coherence = (
            force_velocity_response(force, control_velocity, sample_hz)
        )
        axes_result[axis] = {
            "force_rms": rms(force),
            "force_highpass_rms": hp_rms(force),
            "velocity_raw_rms": rms(raw_velocity),
            "velocity_control_rms": rms(control_velocity),
            "velocity_raw_highpass_rms": hp_rms(raw_velocity),
            "velocity_control_highpass_rms": hp_rms(control_velocity),
            "actual_acceleration_rms": rms(actual_acceleration),
            "admittance_acceleration_rms": rms(admittance_acceleration),
            "actual_jerk_rms": rms(jerk),
            "target_position_rms": rms(target),
            "actual_position_rms": rms(actual),
            "servo_position_rms": rms(servo),
            "tracking_error_rms": rms(tracking),
            "force_dominant_hz": dominant_frequency(force, sample_hz),
            "velocity_dominant_hz": dominant_frequency(
                control_velocity,
                sample_hz,
            ),
            "force_velocity_coherence_frequency_hz": response_frequency,
            "force_velocity_gain_at_max_coherence": response_gain,
            "force_velocity_max_coherence": response_coherence,
        }

    force_xyz = np.column_stack(
        [
            column(
                selected,
                (f"F_ext_{axis}", f"F_filtered_{axis}", f"F_raw_{axis}"),
            )
            for axis in ("x", "y", "z")
        ]
    )
    active = np.linalg.norm(force_xyz, axis=1) > contact_threshold
    active_index = np.flatnonzero(active)
    active_window = None
    if active_index.size:
        active_window = {
            "start_s": float(time[active_index[0]]),
            "end_s": float(time[active_index[-1]]),
            "duration_s": float(
                time[active_index[-1]] - time[active_index[0]]
            ),
        }
    return {
        "file": str(path),
        "rows": len(selected),
        "duration_s": float(time[-1] - time[0]),
        "sample_hz": sample_hz,
        "highpass_hz": highpass_hz,
        "contact_window": active_window,
        "axes": axes_result,
    }


def print_summary(result):
    print(
        f"\n{result['file']}: rows={result['rows']}"
        f" duration={result['duration_s']:.3f}s"
        f" fs={result['sample_hz']:.3f}Hz"
    )
    if result["contact_window"] is not None:
        window = result["contact_window"]
        print(
            "contact="
            f"{window['start_s']:.3f}..{window['end_s']:.3f}s"
        )
    for axis, values in result["axes"].items():
        print(
            f" {axis:>2s}"
            f" F_rms={values['force_rms']:.4g}"
            f" v_raw_rms={values['velocity_raw_rms']:.4g}"
            f" v_ctrl_rms={values['velocity_control_rms']:.4g}"
            f" v_raw_hf={values['velocity_raw_highpass_rms']:.4g}"
            f" v_ctrl_hf={values['velocity_control_highpass_rms']:.4g}"
            f" jerk_rms={values['actual_jerk_rms']:.4g}"
            f" track_rms={values['tracking_error_rms']:.4g}"
            f" F_peak={values['force_dominant_hz']:.3f}Hz"
            f" v_peak={values['velocity_dominant_hz']:.3f}Hz"
            f" coh={values['force_velocity_max_coherence']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze admittance control CSV logs"
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--highpass-hz", type=float, default=5.0)
    parser.add_argument("--contact-threshold", type=float, default=2.0)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    results = [
        analyze_file(
            path,
            args.highpass_hz,
            args.contact_threshold,
            args.start,
            args.end,
        )
        for path in args.csv
    ]
    for result in results:
        print_summary(result)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
