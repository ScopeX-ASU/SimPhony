import math
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, Tuple

__all__ = [
    "calculate_memory_latency_and_energy",
    "generate_memory_setting",
]


def update_cacti_config(
    value: Dict[str, int],
    cacti_path: str = "./cacti",
    cacti_config_file_name: str = "TPUcache.cfg",
):
    # Define the CACTI configuration file path
    # cacti_path = "./cacti"
    cacti_config_file = os.path.join(cacti_path, cacti_config_file_name)

    # Create a temporary file for CACTI configuration updates
    temp_config_file = tempfile.NamedTemporaryFile(
        delete=False, dir="./cacti", suffix=".cfg"
    )
    with open(cacti_config_file, "r") as original_file:
        data_fsram = original_file.readlines()

    # Function to update or insert parameter in the CACTI configuration
    def update_temp_cacti_config(data_fsram, key, val):
        updated = False
        # Define parameter patterns and update them
        patterns = {
            "cache_size": r"^-size \(bytes\) .*",  # regex pattern for cache size
            "bus_width": r"^-output/input bus width .*",  # regex pattern for bus width
            "block_size": r"^-block size \(bytes\) .*",  # regex pattern for block size
        }

        for i, line in enumerate(data_fsram):
            if re.match(patterns[key], line):
                # Update the line if pattern matches
                if key == "cache_size":
                    data_fsram[i] = f"-size (bytes) {int(val)}\n"
                elif key == "bus_width":
                    data_fsram[i] = f"-output/input bus width {int(val)}\n"
                elif key == "block_size":
                    data_fsram[i] = f"-block size (bytes) {int(val)}\n"
                updated = True
                break

        # If the parameter line is not found, add it to the end
        if not updated:
            if key == "cache_size":
                data_fsram.append(f"-size (bytes) {int(val)}\n")
            elif key == "bus_width":
                data_fsram.append(f"-output/input bus width {int(val)}\n")
            elif key == "block_size":
                data_fsram.append(f"-block size (bytes) {int(val)}\n")

        return data_fsram

    # Update all configuration values for this memory type in the temporary file
    for key, val in value.items():
        data_fsram = update_temp_cacti_config(data_fsram, key, val)

    # Write all updates to the temporary configuration file
    with open(temp_config_file.name, "w") as temp_file:
        temp_file.writelines(data_fsram)

    return temp_config_file.name


def run_cacti_simulation(config_file, output_file, memory_type):
    validation_file_name = config_file + ".out"
    config_temp_file = config_file
    try:
        # Run the CACTI simulation
        subprocess.run(
            f"./cacti -infile {config_file} > {output_file}", shell=True, cwd="./cacti"
        )
        print("CACTI simulation completed. Parsing the results...")

        # Check if the provided configuration is valid
        if (
            not os.path.exists(validation_file_name)
            or os.stat(validation_file_name).st_size == 0
        ):
            raise FileNotFoundError(
                f"Output file '{validation_file_name}' does not exist or is empty. Invalid memory configuration provided for {memory_type}."
            )

        # Read and return the CACTI output
        with open(output_file, "r") as file:
            data = file.readlines()

        # Convert the lines list into a single string for easier regex searching
        cacti_output = "".join(data)

    finally:
        # Delete the temporary files
        if os.path.exists(validation_file_name):
            os.remove(validation_file_name)
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(config_temp_file):
            os.remove(config_temp_file)

    return cacti_output


def extract_cycle_time(cacti_output):
    # Regular expression to extract Cycle time
    cycle_time_pattern = r"Cycle time \(ns\):\s+([\d.]+)"
    cycle_time_match = re.search(cycle_time_pattern, cacti_output)

    # Extract and return the cycle time if found
    return float(cycle_time_match.group(1)) if cycle_time_match else float("inf")


def extract_access_times(cacti_output):
    # Regular expression to extract Access time
    access_time_pattern = r"Access time \(ns\):\s+([\d.]+)"

    access_time_match = re.search(access_time_pattern, cacti_output)

    # Extract and return the access times if found
    access_time = float(access_time_match.group(1)) if access_time_match else None

    return access_time


def extract_dynamic_read_energy(cacti_output):
    """
    Extracts the total dynamic read energy per access from CACTI output.
    """
    read_energy_pattern = r"Total dynamic read energy per access \(nJ\):\s+([\d.]+)"
    read_energy_match = re.search(read_energy_pattern, cacti_output)
    return float(read_energy_match.group(1)) if read_energy_match else None


def extract_dynamic_write_energy(cacti_output):
    """
    Extracts the total dynamic write energy per access from CACTI output.
    """
    write_energy_pattern = r"Total dynamic write energy per access \(nJ\):\s+([\d.]+)"
    write_energy_match = re.search(write_energy_pattern, cacti_output)
    return float(write_energy_match.group(1)) if write_energy_match else None


def extract_data_array_area(cacti_output):
    """
    Extracts the data array area from CACTI output.
    """
    area_pattern = r"Data array: Area \(mm2\):\s+([\d.]+)"
    area_match = re.search(area_pattern, cacti_output)
    return float(area_match.group(1)) if area_match else None


def extract_data_array_height(cacti_output):
    """
    Extracts the data array height from CACTI output.
    """
    height_pattern = r"Height \(mm\):\s+([\d.]+)"
    height_match = re.search(height_pattern, cacti_output)
    return float(height_match.group(1)) if height_match else None


def extract_data_array_width(cacti_output):
    """
    Extracts the data array width from CACTI output.
    """
    width_pattern = r"Width \(mm\):\s+([\d.]+)"
    width_match = re.search(width_pattern, cacti_output)
    return float(width_match.group(1)) if width_match else None


def extract_leakage_power(cacti_output):
    """
    Extracts the total leakage power of a bank from CACTI output.
    """
    leakage_power_pattern = r"Total leakage power of a bank \(mW\):\s+([\d.]+)"
    leakage_power_match = re.search(leakage_power_pattern, cacti_output)
    return float(leakage_power_match.group(1)) if leakage_power_match else None


def extract_gate_leakage_power(cacti_output):
    """
    Extracts the total gate leakage power of a bank from CACTI output.
    """
    gate_leakage_power_pattern = (
        r"Total gate leakage power of a bank \(mW\):\s+([\d.]+)"
    )
    gate_leakage_power_match = re.search(gate_leakage_power_pattern, cacti_output)
    return (
        float(gate_leakage_power_match.group(1)) if gate_leakage_power_match else None
    )


def calculate_required_memory_bandwidth(
    arch_dims: Dict[str, int],
    dim_map: Dict[str, str],
    in_bits: int,
    w_bits: int,
    out_bits: int,
    work_freq: int,
    memory_type: str,
    max_layer_size: float | int,
) -> float:
    partition_N = arch_dims[dim_map["N"]] if dim_map["N"] is not None else 1
    partition_M = arch_dims[dim_map["M"]] if dim_map["M"] is not None else 1
    partition_D = arch_dims[dim_map["D"]] if dim_map["D"] is not None else 1

    # One cycle input, weight, and output data under current working frequency
    # work freq in GHz scale, work_freq in GHz, 1e9 convert to Hz
    # Required bandwidth in bits/s
    if memory_type == "RF" or memory_type == "GLB1":
        required_bandwidth = (
            partition_N * partition_M * out_bits
            + partition_N * partition_D * w_bits
            + partition_M * partition_D * in_bits
        ) * (work_freq * 1e9)

    elif memory_type == "GLB2":
        required_bandwidth = (
            max_layer_size
            * (work_freq * 1e9)
            / (partition_D * partition_M * partition_N)
        )

    return required_bandwidth


def generate_memory_setting(
    sub_arch: Dict[str, Any],
    model_partition_cycles: Dict[str, Dict[str, float]],
    layer_sizes: Dict[str, Dict[str, float]],
    # requried_bandwidth_dict: Dict[str, float],
    arch_to_dim_map: Dict[str, Dict[str, int]],
    dim_map: Dict[str, Dict[str, int]],
    # # Below are determined by dataflow, it will calculate the largest required bandwidth and put corresponding architecture configuration here
):
    memory_config = sub_arch["memory"]
    work_freq = sub_arch["core"]["work_freq"]
    dataflow = sub_arch["core"].get("dataflow", "weight_stationary")

    # bits/second
    # required_bandwidth = max(requried_bandwidth_dict.values())

    largest_total_size_layer = max(
        layer_sizes, key=lambda x: layer_sizes[x]["total_size"]
    )
    # Bytes
    max_cache_size = layer_sizes[largest_total_size_layer]["total_size"]

    # Bytes
    max_input_size = layer_sizes[largest_total_size_layer]["input_size"][0]
    input_precision = layer_sizes[largest_total_size_layer]["input_size"][1]

    # Bytes
    max_weight_size = layer_sizes[largest_total_size_layer]["weight_size"][0]
    weight_precision = layer_sizes[largest_total_size_layer]["weight_size"][1]

    # Bytes
    max_output_size = layer_sizes[largest_total_size_layer]["output_size"][0]
    output_precision = layer_sizes[largest_total_size_layer]["output_size"][1]

    iter_N, iter_D, iter_M, _, _, _, _, _ = model_partition_cycles[
        largest_total_size_layer
    ]

    # Bits/ second

    memory_simulation_results = {}

    for memory_type, memory_params in memory_config.items():
        # Not simulating "HBM" in cacti
        if memory_type != "HBM":
            block_size = memory_params["block_size"]
            config_file = memory_params["config_file_name"]
            output_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", dir="./cacti"
            ).name

            template_cache_size = memory_params["cache_size"]  # Bytes

            # Block size in bytes
            # bus width in bits
            # Thus bus width = block size * 8
            bus_width = block_size * 8  # Bits

            required_bandwidth = calculate_required_memory_bandwidth(
                arch_dims=arch_to_dim_map,
                dim_map=dim_map[largest_total_size_layer],
                in_bits=input_precision,
                w_bits=weight_precision,
                out_bits=output_precision,
                work_freq=work_freq,
                memory_type=memory_type,
                max_layer_size=max_cache_size,
            )

            # Bytes
            # GLB2 cache size should be able to contain the model's largest layer
            if memory_type == "GLB2":
                required_cache_size = max_cache_size
            # GLB1 should contains the data that uses for entire inner loop iteration
            elif memory_type == "GLB1":
                # partition size is the smallest partition size for photonic tensor cores under one cycle
                partition_size_N = (
                    arch_to_dim_map[dim_map[largest_total_size_layer]["N"]]
                    if dim_map[largest_total_size_layer]["N"] is not None
                    else 1
                )
                partition_size_D = (
                    arch_to_dim_map[dim_map[largest_total_size_layer]["D"]]
                    if dim_map[largest_total_size_layer]["D"] is not None
                    else 1
                )
                partition_size_M = (
                    arch_to_dim_map[dim_map[largest_total_size_layer]["M"]]
                    if dim_map[largest_total_size_layer]["M"] is not None
                    else 1
                )
                input_partition_size = partition_size_D * partition_size_M  # Bytes
                weight_partition_size = partition_size_N * partition_size_D  # Bytes
                output_partition_size = partition_size_N * partition_size_M  # Bytes

                # E.g. for weight stationary,
                # for n in iter_N:
                #   for d in iter_D:
                #       for m in iter_M:
                # the cache should contain the data for all iter_M:
                # Then load again for the next iter_D
                if dataflow == "weight_stationary":
                    required_cache_size = (
                        input_partition_size * iter_M
                        + weight_partition_size
                        + output_partition_size * iter_M
                    )
                # E.g. for output stationary,
                # for n in iter_N:
                #   for m in iter_M:
                #       for d in iter_D:
                # the cache should contain the data for all iter_D:
                # Then load again for the next iter_M
                elif dataflow == "output_stationary":
                    required_cache_size = (
                        input_partition_size * iter_D
                        + weight_partition_size * iter_D
                        + output_partition_size
                    )
                # E.g. for input stationary,
                # for m in iter_M:
                #   for d in iter_D:
                #       for n in iter_N:
                # the cache should contain the data for all iter_N:
                # Then load again for the next iter_D
                elif dataflow == "input_stationary":
                    required_cache_size = (
                        input_partition_size
                        + weight_partition_size * iter_N
                        + output_partition_size * iter_N
                    )

            # RF should contains the data of input, weight, and output for one cycle
            elif memory_type == "RF":
                required_cache_size = (
                    input_partition_size + weight_partition_size + output_partition_size
                )

            memory_simulation_results[memory_type] = {
                "block_size": memory_params["block_size"],
                "cache_size": template_cache_size,
                "bus_width": bus_width,
            }

            temp_config_file = update_cacti_config(
                memory_simulation_results[memory_type],
                cacti_path="./configs/memory",
                cacti_config_file_name=config_file,
            )

            cacti_output = run_cacti_simulation(
                temp_config_file, output_file, memory_type
            )

            real_cycle_time = extract_cycle_time(cacti_output)

            real_cycle_time_s = real_cycle_time / 1e9  # ns to s

            bandwidth_per_module = bus_width / real_cycle_time_s  # bits/second

            # Cacularate the required memory number based on real cycle time and required bandwidth
            required_memory_number = math.ceil(
                required_bandwidth / bandwidth_per_module
            )

            real_bandwidth = bus_width / (real_cycle_time_s / 1e9) / 8  # Bytes/second

            # Calculate the final cache size based on the required memory number
            final_sram_cache = template_cache_size * required_memory_number

            # Update the cache size in the memory simulation results
            if (
                final_sram_cache >= required_cache_size
            ):  # If the final cache size is larger than the required cache size
                memory_simulation_results[memory_type][
                    "total_cache_size"
                ] = final_sram_cache
            else:  # If the final cache size is smaller than the required cache size, increase the cache size to the next multiple of the template cache size
                temp_required_size = math.ceil(
                    required_cache_size / template_cache_size
                )
                memory_simulation_results[memory_type]["total_cache_size"] = (
                    template_cache_size * temp_required_size
                )

            access_time = extract_access_times(cacti_output)
            gate_leakage_power = extract_gate_leakage_power(cacti_output)  # mW
            leakage_power = extract_leakage_power(cacti_output)  # mW
            area = extract_data_array_area(cacti_output)  # mm^2
            height = extract_data_array_height(cacti_output)  # mm
            width = extract_data_array_width(cacti_output)  # mm
            dynamic_read_energy = (
                extract_dynamic_read_energy(cacti_output) * 1000
            )  # nJ to pJ
            dynamic_write_energy = (
                extract_dynamic_write_energy(cacti_output) * 1000
            )  # nJ to pJ

            memory_simulation_results[memory_type].update(
                {
                    "bandwidth": real_bandwidth,  # Bytes/second
                    "cycle_time": real_cycle_time / block_size,  # ns/byte
                    "access_time": access_time / block_size,  # ns/byte
                    "number_of_srams": required_memory_number,
                    "read_energy": dynamic_read_energy,  # pJ
                    "write_energy": dynamic_write_energy,  # pJ
                    "area": area,  # mm^2
                    "height": height,  # mm
                    "width": width,  # mm
                    "gate_leakage_power": gate_leakage_power,  # mW
                    "leakage_power": leakage_power,  # mW
                }
            )
        else:
            # HBM is not simulated in CACTI
            memory_simulation_results[memory_type] = {
                "bandwidth": memory_params["bandwidth"] * 1e9 * 8,  # bits/second
                "read_energy": memory_params["read_energy"],  # pJ
                "write_energy": memory_params["write_energy"],  # pJ
            }
    return memory_simulation_results


def calculate_memory_latency_and_energy(
    memory_simulation_results: Dict[str, Dict[str, Any]],
    partition_cycles: Dict[str, Dict[str, float]],
    layer_sizes: Dict[str, Dict[str, float]],
    layer_to_sub_arch_mapping: Dict[str, Dict[str, Any]],
    arch_to_dim_map: Dict[str, Dict[str, int]],
    dim_map: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    layer_memory_loading_latency = {}
    layer_memory_loading_energy = {}
    memory_total_energy = {"HBM": 0, "GLB2": 0, "GLB1": 0, "RF": 0}

    # layer_num = len(layer_sizes)

    for layer_name, layer_size in layer_sizes.items():
        # layer_total_size = layer_size["total_size"]  # Bytes

        GLB2_input_size = layer_size["input_size"][0]  # Bytes
        GLB2_weight_size = layer_size["weight_size"][0]  # Bytes
        GLB2_output_size = layer_size["output_size"][0]  # Bytes

        iter_N, iter_D, iter_M, _, _, _, _, _ = partition_cycles[layer_name]

        partition_size_N = (
            arch_to_dim_map[dim_map[layer_name]["N"]]
            if dim_map[layer_name]["N"] is not None
            else 1
        )
        partition_size_D = (
            arch_to_dim_map[dim_map[layer_name]["D"]]
            if dim_map[layer_name]["D"] is not None
            else 1
        )
        partition_size_M = (
            arch_to_dim_map[dim_map[layer_name]["M"]]
            if dim_map[layer_name]["M"] is not None
            else 1
        )

        input_partition_size = partition_size_D * partition_size_M  # Bytes
        weight_partition_size = partition_size_N * partition_size_D  # Bytes
        output_partition_size = partition_size_N * partition_size_M  # Bytes

        dataflow = layer_to_sub_arch_mapping[layer_name]["dataflow"]
        temporal_factor_config = layer_to_sub_arch_mapping[layer_name]["core"][
            "netlist"
        ]["temporal_accum_factor"]
        temporal_duration = temporal_factor_config["duration"]
        temporal_type = temporal_factor_config["memory"]
        # Only support temporal accumulation at the output
        if temporal_type is not None:
            output_factor = temporal_duration
        else:
            output_factor = 1

        if dataflow == "weight_stationary":
            GLB1_input_size = input_partition_size * iter_M  # Bytes
            GLB1_weight_size = weight_partition_size  # Bytes
            GLB1_output_size = output_partition_size * iter_M  # Bytes
        elif dataflow == "output_stationary":
            GLB1_input_size = input_partition_size * iter_D  # Bytes
            GLB1_weight_size = weight_partition_size * iter_D  # Bytes
            GLB1_output_size = output_partition_size  # Bytes
        elif dataflow == "input_stationary":
            GLB1_input_size = input_partition_size  # Bytes
            GLB1_weight_size = weight_partition_size * iter_N  # Bytes
            GLB1_output_size = output_partition_size * iter_N  # Bytes

        RF_input_size = input_partition_size  # Bytes
        RF_weight_size = weight_partition_size  # Bytes
        RF_output_size = output_partition_size  # Bytes

        if dataflow == "weight_stationary":
            RF_weight_iteration = iter_N * iter_D
            RF_input_iteration = RF_output_iteration = iter_M * iter_N * iter_D

            GLB_1_weight_iterations = GLB_1_input_iterations = iter_D * iter_N

            GLB_1_output_iterations = iter_N
        elif dataflow == "input_stationary":
            RF_input_iteration = iter_M * iter_D
            RF_weight_iteration = RF_output_iteration = iter_M * iter_N * iter_D

            GLB_1_weight_iterations = GLB_1_input_iterations = iter_M * iter_D
            GLB_1_output_iterations = iter_M

        elif dataflow == "output_stationary":
            RF_output_iteration = iter_N * iter_M
            RF_input_iteration = RF_weight_iteration = iter_N * iter_D * iter_M

            GLB_1_weight_iterations = GLB_1_input_iterations = iter_N * iter_M
            GLB_1_output_iterations = iter_N * iter_M

        # Level 1 input energy
        # memory_simulation_results["GLB1"]["read_energy"] * (GLB1_input_size + GLB2_weight_size) * GLB1_input_loading_iterations +
        #
        # memory_total_energy["GLB2"] +=
        memory_total_energy["HBM"] += memory_simulation_results["HBM"][
            "write_energy"
        ] * (
            GLB2_output_size
        ) + memory_simulation_results[  # Output data from GLB2 to HBM
            "HBM"
        ][
            "read_energy"
        ] * (
            GLB2_input_size + GLB2_weight_size
        )  # Input data from HBM to GLB2

        memory_total_energy["GLB2"] += (
            memory_simulation_results["GLB2"]["write_energy"]
            * (GLB2_input_size + GLB2_weight_size)  # Input data from HBM to GLB2
            + memory_simulation_results["GLB2"]["read_energy"]
            * (
                GLB1_input_size * GLB_1_input_iterations
                + GLB1_weight_size * GLB_1_weight_iterations
            )  # Input data from GLB2 to GLB1
            + memory_simulation_results["GLB2"]["write_energy"]
            * (
                GLB1_output_size * GLB_1_output_iterations
            )  # Output data from GLB1 to GLB2
            + memory_simulation_results["GLB2"]["read_energy"]
            * (GLB2_output_size)  # Output data from GLB2 to HBM
        )

        memory_total_energy["GLB1"] += (
            memory_simulation_results["GLB1"]["write_energy"]
            * (
                GLB1_input_size * GLB_1_input_iterations
                + GLB1_weight_size * GLB_1_weight_iterations
            )  # Input data from GLB2 to GLB1
            + memory_simulation_results["GLB1"]["read_energy"]
            * (
                RF_input_size * RF_input_iteration
                + RF_weight_iteration * RF_weight_size
            )  # Input data from GLB1 to RF
            + memory_simulation_results["GLB1"]["write_energy"]
            * (RF_output_iteration * RF_output_size)  # Output data from RF to GLB1
            + memory_simulation_results["GLB1"]["read_energy"]
            * (
                GLB1_output_size * GLB_1_output_iterations
            )  # Output data from GLB1 to GLB2
        )

        memory_total_energy["RF"] += (
            memory_simulation_results["RF"]["write_energy"]
            * (
                RF_input_size * RF_input_iteration
                + RF_weight_iteration * RF_weight_size
            )  # Input data from GLB1 to RF
            + memory_simulation_results["RF"]["read_energy"]
            * (RF_output_iteration * RF_output_size)
            / output_factor  # Output data from RF to GLB1
        ) * 2  # RF considers both the data from/to GLB1, and the data from ADC and to DAC

        # PTCim: The first open source, data dependent, and scalable photonic tensor core simulator

        GLB2_input_latency = (
            GLB2_input_size + GLB2_weight_size
        ) / memory_simulation_results["GLB2"][
            "bandwidth"
        ]  # Bytes / Bytes/second = second
        GLB2_output_latency = (
            GLB2_output_size / memory_simulation_results["GLB2"]["bandwidth"]
        )  # Bytes / Bytes/second = second

        GLB1_input_latency = (
            GLB1_input_size + GLB1_weight_size
        ) / memory_simulation_results["GLB1"][
            "bandwidth"
        ]  # Bytes / Bytes/second = second
        GLB1_output_latency = (
            GLB1_output_size / memory_simulation_results["GLB1"]["bandwidth"]
        )  # Bytes / Bytes/second = second

        RF_input_latency = (RF_input_size + RF_weight_size) / memory_simulation_results[
            "RF"
        ][
            "bandwidth"
        ]  # Bytes / Bytes/second = second
        RF_output_latency = (
            RF_output_size / memory_simulation_results["RF"]["bandwidth"]
        )  # Bytes / Bytes/second = second

        layer_memory_loading_latency[layer_name] = {
            "GLB2_input_latency": GLB2_input_latency,
            "GLB2_output_latency": GLB2_output_latency,
            "GLB1_input_latency": GLB1_input_latency,
            "GLB1_output_latency": GLB1_output_latency,
            "RF_input_latency": RF_input_latency,
            "RF_output_latency": RF_output_latency,
        }
        # print(layer_memory_loading_latency)
        # exit(0)
        layer_memory_loading_energy[layer_name] = memory_total_energy

    return layer_memory_loading_latency, memory_total_energy
