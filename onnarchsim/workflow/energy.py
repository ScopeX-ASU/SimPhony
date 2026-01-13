import math
from typing import Any, Dict, List, Optional, Tuple

from onnarchsim.workflow.utils import (
    evaluate_factors,
    load_device_and_count,
    load_required_devices,
    parse_device_name,
)

__all__ = [
    "reuse_determinator",
    "load_device_and_count",
    "load_required_devices",
    "initalize_all_devices_power",
    "energy_calculator",
    "chip_energy_calculator",
]


def cal_DAC_power(
    config: Dict[str, List[str]],
    in_bit: int,
    work_freq: int,
) -> float:
    # convert power to desired freq and bit width
    assert (
        in_bit <= config["cfgs"]["prec"]
    ), f"Got input bit {in_bit} exceeds the DAC precision limit"
    if config["cfgs"]["FoM"] is not None:
        # following 2 * FoM * nb * Fs / Br (assuming Fs=Br)
        DAC_power = 2 * config["cfgs"]["FoM"] * in_bit * work_freq * 1e-3
    else:
        # P \propto 2**N/(N+1) * f_clk
        DAC_power = (
            config["dynamic_power"]
            * (2**in_bit / (in_bit + 1))
            / (2 ** config["cfgs"]["prec"] / (config["cfgs"]["prec"] + 1))
            * work_freq
            / config["cfgs"]["sample_rate"]
        )

    return DAC_power


def cal_ADC_power(
    config: Dict[str, List[str]],
    out_bit: int,
    work_freq: int,
) -> float:
    # convert power to desired freq and bit width
    assert (
        out_bit <= config["cfgs"]["prec"]
    ), f"Got input bit {out_bit} exceeds the DAC precision limit"
    if config["cfgs"]["type"] == "sar":
        # P \propto N
        ADC_power = (
            config["dynamic_power"]
            * work_freq
            / config["cfgs"]["sample_rate"]
            * (out_bit / config["cfgs"]["prec"])
        )

    elif config["cfgs"]["type"] == "flash":
        # P \propto (2**N - 1)
        ADC_power = (
            config["dynamic_power"]
            * work_freq
            / config["cfgs"]["sample_rate"]
            * ((2**out_bit - 1) / (2 ** config["cfgs"]["prec"] - 1))
        )
    else:
        raise NotImplementedError(f"ADC type {config['cfgs']['type']} not supported")

    return ADC_power


def cal_MZM_power(
    config: Dict[str, List[str]],
    in_bit: int,
    work_freq: int,
) -> float:
    # GHz
    max_symbol_rate = config["cfgs"]["modulation_speed"] / config["cfgs"]["testing_bit"]
    assert (
        max_symbol_rate >= work_freq
    ), f"Got work freq {work_freq} GHz exceeds max symbol rate {max_symbol_rate} GHz"
    # Now return the fj/switch tested at testing bit mW
    #  fj/bit * bit * GHz * 1e-3      mW
    return (
        config["cfgs"]["efficiency"] * work_freq * 1e-3 * config["cfgs"]["testing_bit"]
    )


def cal_PS_power(
    config: Dict[str, List[str]],
    work_freq: int,
) -> float:
    return config["cfgs"]["P_pi"]


def cal_laser_power(
    laser_config: Dict[str, List[str]],
    in_bit: int,
    insertion_loss: float,
    photo_detector_sensitivity: float,
    modulation_extinction_ratio: float,
    distribution_factor: int = 1,
) -> float:
    wall_plug_efficiency = laser_config["cfgs"]["wall_plug_eff"]
    P_laser_dbm = (
        insertion_loss
        + photo_detector_sensitivity
        + 10 * math.log10(distribution_factor)
    )
    laser_power = 10 ** (P_laser_dbm / 10) / wall_plug_efficiency * 2**in_bit
    extinction_scale = 1 / (1 - 0.1 ** (modulation_extinction_ratio / 10))
    laser_power = laser_power * extinction_scale
    return laser_power


def initalize_all_devices_power(
    sub_arch: Dict[str, Any],
    components: Dict[str, Any],
    insertion_loss: float,
) -> Dict[str, Any]:
    R = sub_arch.get("tiles", None)
    C = sub_arch.get("cores_per_tile", None)
    core_config = sub_arch.get("core", {})
    H = core_config.get("height", 1)
    W = core_config.get("width", 1)
    N = core_config.get("num_wavelength", 1)
    work_freq = core_config.get("work_freq", 3)
    extinction_ratio = 0
    sensitivity = -27  # Default value
    laser_power_distribution_factor = evaluate_factors(
        core_config.get("netlist", {}).get("laser_power_distribution_factor", 1),
        R,
        C,
        H,
        W,
        N,
    )
    in_bit = core_config.get("precision", {}).get("in_bit", 8)
    out_bit = core_config.get("precision", {}).get("out_bit", 8)

    # Laser config reference
    laser_config = {}

    for device, device_cfg in components.items():
        part, type, index, instance, name = parse_device_name(device)
        if "adc" in type:
            dynamic_power = cal_ADC_power(device_cfg, out_bit, work_freq)
            device_cfg["dynamic_power"] = dynamic_power
        elif "dac" in type:
            dynamic_power = cal_DAC_power(device_cfg, in_bit, work_freq)
            device_cfg["dynamic_power"] = dynamic_power
        elif "mzm" in type:
            dynamic_power = cal_MZM_power(device_cfg, in_bit, work_freq)

            current_extinction_ratio = (
                device_cfg["cfgs"]["extinction_ratio"]
                if device_cfg["cfgs"]["extinction_ratio"] is not None
                else 0
            )
            extinction_ratio = (
                current_extinction_ratio
                if current_extinction_ratio > extinction_ratio
                else extinction_ratio
            )
            device_cfg["dynamic_power"] = dynamic_power
        elif "phase_shifter" in type:
            dynamic_power = cal_PS_power(device_cfg, work_freq)
            device_cfg["tuning_cycles"] = (
                device_cfg["cfgs"]["response_time"] * work_freq
            )
            device_cfg["dynamic_power"] = dynamic_power
        elif "photodetector" in type:
            current_sensitivity = device_cfg["cfgs"]["sensitivity"]
            sensitivity = (
                current_sensitivity
                if current_sensitivity > sensitivity
                else sensitivity
            )
        elif "on_chip_laser" in type or "off_chip_laser" in device:
            laser_config = device_cfg
            continue
        # Default values, if not specificly defined, then set to 0 for power and 1 for tuning cycles
        if "dynamic_power" not in device_cfg:
            device_cfg["dynamic_power"] = 0
        if "static_power" not in device_cfg:
            device_cfg["static_power"] = 0
        if "tuning_cycles" not in device_cfg:
            device_cfg["tuning_cycles"] = 1
        for key in list(device_cfg.keys()):
            if key not in [
                "count",
                "static_power",
                "dynamic_power",
                "tuning_cycles",
                "chip_type",
            ]:
                device_cfg.pop(key)

        # Energy pJ per switch  Config Power is mW, work freq is GHz
        device_cfg["static_energy"] = device_cfg["static_power"] / work_freq
        device_cfg["dynamic_energy"] = device_cfg["dynamic_power"] / work_freq

    # for device, device_cfg in components.items():
    #     if "off_chip_laser" in device or "on_chip_laser" in device:

    static_power = cal_laser_power(
        laser_config,
        in_bit,
        insertion_loss,
        sensitivity,
        extinction_ratio,
        laser_power_distribution_factor,
    )
    laser_config["dynamic_power"] = 0
    laser_config["static_power"] = static_power
    laser_config["tuning_cycles"] = 1
    laser_config["static_energy"] = static_power / work_freq
    laser_config["dynamic_energy"] = 0

    for key in list(laser_config.keys()):
        if key not in [
            "count",
            "static_power",
            "dynamic_power",
            "tuning_cycles",
            "chip_type",
        ]:
            laser_config.pop(key)

    # Energy pJ per switch  Config Power is mW, work freq is GHz
    laser_config["static_energy"] = laser_config["static_power"] / work_freq
    laser_config["dynamic_energy"] = laser_config["dynamic_power"] / work_freq

    return components


def reuse_determinator(
    # dim_map: Dict[str, Optional[str]],
    dataflow: str,
    iter_M: int,
    iter_N: int,
    iter_D: int,
    max_operand_1_tuning_cycles: int,
    max_operand_2_tuning_cycles: int,
    max_tuning_cycles: int,
    forward_factor_w: int,
    forward_factor_x: int,
) -> Tuple[int, int, int]:
    # Inner loop is M, Outer loop can be ND or DN
    if dataflow == "weight_stationary":
        inner_loop = max_operand_1_tuning_cycles
        outer_loop = max_tuning_cycles
        operand_1_reuse = 1
        operand_2_reuse = iter_M

        max_cycle = ((iter_M * iter_N * iter_D) * inner_loop) + (
            iter_N * iter_D * outer_loop
        )

    # Inner loop is N, Outer loop can be MD or DM
    elif dataflow == "input_stationary":
        inner_loop = max_operand_2_tuning_cycles
        outer_loop = max_tuning_cycles
        operand_1_reuse = iter_N
        operand_2_reuse = 1

        max_cycle = ((iter_M * iter_N * iter_D) * inner_loop) + (
            iter_M * iter_D * outer_loop
        )

    # Inner loop is D, Outer loop can be MN or NM
    elif dataflow == "output_stationary":
        inner_loop = max_tuning_cycles
        outer_loop = max_tuning_cycles

        operand_1_reuse = 1
        operand_2_reuse = 1

        max_cycle = ((iter_M * iter_N * iter_D) * inner_loop) + (
            iter_M * iter_N * outer_loop
        )

    else:
        raise ValueError(f"Dataflow {dataflow} not supported")

    operand_2_reuse = operand_2_reuse * forward_factor_w
    operand_1_reuse = operand_1_reuse * forward_factor_x
    max_cycle = max_cycle * forward_factor_w * forward_factor_x

    return 1 / operand_2_reuse, 1 / operand_1_reuse, max_cycle


def energy_calculator(
    sub_arch: Dict[str, Any],
    device_db: Dict[str, Any],
    dim_map_cfgs: Dict[str, Optional[str]],
    cycles: List[Tuple[int, int, int, int, int, int]],
    insertion_loss: float,
) -> Dict[str, Any]:
    # Get dataflow from config
    dataflow = sub_arch.get("dataflow", "weight_stationary")
    # Get frequency from config
    work_freq = sub_arch["core"]["work_freq"]

    # instance_config = sub_arch["core"]["netlist"].get("instance", {})

    components, operand_1_devices, operand_2_devices, temporal_accum_devices = (
        load_required_devices(sub_arch, device_db)
    )

    # Here is poping the node instance where it is used for area calculation,
    # but not for energy calculation
    # TODO: Add support for multiple nodes per core
    components.pop("node")

    # Initalize all devices' power and count
    # This will return a dictionary of devices with their power and count
    # All devices declared in the instance will be initialized, including both core and node devices
    # TODO: Add support for multiple nodes per core
    devices = initalize_all_devices_power(sub_arch, components, insertion_loss)

    # If operand devices are not found, raise an error
    # We cannot make a fair assumption on device tuning cycles if user does not provide which devices is used for operand encoding
    assert (
        len(operand_1_devices) > 0
    ), "No operand 1 devices found in the architecture config, cannot make fair assumption on device tuning cycles"

    assert (
        len(operand_2_devices) > 0
    ), "No operand 2 devices found in the architecture config, cannot make fair assumption on device tuning cycles"

    # Initialize a dictionary for computation latency report
    total_computation_latency = {"total_tuning_latency": 0, "total_static_latency": 0}

    max_operand_1_tuning_cycles = max(
        [devices[device]["tuning_cycles"] for device in operand_1_devices]
    )
    max_operand_2_tuning_cycles = max(
        [devices[device]["tuning_cycles"] for device in operand_2_devices]
    )
    max_tuning_cycles = max(max_operand_1_tuning_cycles, max_operand_2_tuning_cycles)

    # Start to process each layer in layers related to one sub-architecutre
    for layer_name, (
        iter_N,
        iter_D,
        iter_M,
        N,
        D,
        M,
        forward_factor_w,
        forward_factor_x,
    ) in cycles.items():
        # Calculate the total switching cycles for this layer
        # This is the total number of cycles needed to encoding all the data
        # Difference is refected on dataflow, but not on the total number of cycles
        switching_cycles = iter_N * iter_D * iter_M

        operand_2_duty_cycles, operand_1_duty_cycles, max_cycle = reuse_determinator(
            dataflow,
            iter_M,
            iter_N,
            iter_D,
            max_operand_1_tuning_cycles,
            max_operand_2_tuning_cycles,
            max_tuning_cycles,
            forward_factor_w,
            forward_factor_x,
        )

        for device, device_cfg in devices.items():
            # part, type, index, instance, name = parse_device_name(device)

            # pJ per switch * switching cycles
            total_dynamic_energy = (
                device_cfg["dynamic_energy"] * switching_cycles * device_cfg["count"]
            )
            # For temporal shared devices, we need to divide the total dynamic energy by the temporal duration
            duration = temporal_accum_devices.get(device, None)
            total_dynamic_energy = (
                total_dynamic_energy / duration
                if duration is not None
                else total_dynamic_energy
            )

            total_dynamic_energy = (
                total_dynamic_energy * operand_1_duty_cycles
                if device in operand_1_devices
                else total_dynamic_energy
            )
            total_dynamic_energy = (
                total_dynamic_energy * operand_2_duty_cycles
                if device in operand_2_devices
                else total_dynamic_energy
            )

            # pJ per cycle * max cycle (No memory bandwidth limit)
            total_static_energy = (
                device_cfg["static_energy"] * max_cycle * device_cfg["count"]
            )

            if "total_energy" not in device_cfg:
                device_cfg["total_dynamic_energy"] = total_dynamic_energy
                device_cfg["total_static_energy"] = total_static_energy
                device_cfg["total_energy"] = total_dynamic_energy + total_static_energy
            else:
                device_cfg["total_dynamic_energy"] += total_dynamic_energy
                device_cfg["total_static_energy"] += total_static_energy
                device_cfg["total_energy"] += total_dynamic_energy + total_static_energy

        total_computation_latency["total_tuning_latency"] += switching_cycles
        total_computation_latency["total_static_latency"] += max_cycle

    total_computation_latency["total_tuning_latency"] = (
        total_computation_latency["total_tuning_latency"] / work_freq * 10e-9
    )
    total_computation_latency["total_static_latency"] = (
        total_computation_latency["total_static_latency"] / work_freq * 10e-9
    )

    return devices, total_computation_latency


def chip_energy_calculator(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Initialize dictionaries to classify nodes by chip type
    chip_classification = {}
    for device, energy_config in configs.items():
        chip_type = energy_config.get("chip_type", "off_chip")

        # Initialize the entry for this chip type if not already present
        if chip_type not in chip_classification:
            chip_classification[chip_type] = {
                "devices": [device],
                "total_dynamic_energy": energy_config["total_dynamic_energy"],
                "total_static_energy": energy_config["total_static_energy"],
            }
        else:
            chip_classification[chip_type]["devices"].append(device)
            chip_classification[chip_type]["total_dynamic_energy"] += energy_config[
                "total_dynamic_energy"
            ]
            chip_classification[chip_type]["total_static_energy"] += energy_config[
                "total_static_energy"
            ]

    return chip_classification
