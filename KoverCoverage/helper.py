import glob
import json
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import xmltodict


def parse_xml(xml_path) -> dict[str, int]:
    with open(xml_path) as file:
        data = xmltodict.parse(file.read())
        res = {x['@type']: x['@covered'] for x in data['report']['counter']}
    print(res)
    return res


def save_ic(output):
    command = [
        "cp",
        "KoverCoverage/run/kover-kotlin-report.ic",
        os.path.join(output, "last-kover-kotlin-report.ic")
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Произошла ошибка при выполнении команды:", e)
        print(e.stderr)


def move_ic(fr, to):
    command = [
        "cp",
        fr,
        to
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Произошла ошибка при выполнении команды:")
        print(e.stderr)


def create_directory(path: str):
    """Creates a directory at the specified path, including any intermediate directories."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' successfully created.")
    except Exception as e:
        print(f"An error occurred while creating directory '{path}': {e}")


def delete(path: str):
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Error occured while deleting: {e}")


def save_experiment_state(iteration, graphs, x_axis, file_name):
    state = {
        'iteration': iteration,
        'graphs': dict(graphs),
        'x_axis': x_axis
    }
    with open(file_name, 'w') as f:
        json.dump(state, f, indent=4)


def load_experiment_state(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"{file_name} not found")


def restore_experiment_state(param: str, json_dir: str):
    try:
        state = load_experiment_state(json_dir)
        return state[param]
    except FileNotFoundError as e:
        print(e)


def check_ic(file_path, reference_path):
    # Проверяем, существует ли файл
    reference_size = 0
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    # Получаем размер файла
    file_size = os.path.getsize(file_path)
    if os.path.isfile(reference_path):
        reference_size = os.path.getsize(reference_path)

    print(f'File size is: {file_size}\n')
    # Возвращаем True, если размер файла больше нуля, иначе False
    # return file_size >= reference_size
    return file_size > 0


def merge_ic(output: str):
    # report_paths = []
    # pattern = os.path.join(output, "kover-kotlin-report-*.ic")
    # report_paths.extend(glob.glob(pattern))

    command = [
        "java",
        "-jar",
        "KoverCoverage/Kover/sources/kover-cli-0.9.0-SNAPSHOT.jar",
        "merge",
        os.path.join(output, "merged-kover-kotlin-report.ic"),
        os.path.join(output, "last-kover-kotlin-report.ic"),
        "--target",
        os.path.join(output, "merged-kover-kotlin-report.ic"),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Произошла ошибка при выполнении команды:")
        print(e.stderr)


def build_graphs(graphs, x_axis):
    for key, values in graphs.items():
        fig, ax = plt.subplots()
        ax.plot(x_axis[1:], values[1:], label=key)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'{key} Coverage')
        ax.legend()

        plt.show()


def cleanup(target_dir):
    os.environ["JAVA_OPTS"] = ""
    delete(target_dir)
    delete("KoverCoverage/run/kover-kotlin-report.ic")
    delete("KoverCoverage/run/data.xml")


def prepare(compiler, target_dir, json_file, output):
    delete("KoverCoverage/run/backup")
    create_directory("KoverCoverage/run/backup")

    delete("KoverCoverage/run/kover-kotlin-report.ic")
    delete("KoverCoverage/run/data.xml")
    delete(target_dir)

    if not os.path.isdir(compiler):
        print(f"given '{compiler}' doesn't exist")
    else:
        try:
            shutil.copytree(compiler, target_dir)
        except Exception as e:
            print(f"Error occurred during copying of target: {e}")

    merged_report_path = os.path.join(output, "merged-kover-kotlin-report.ic")

    # Проверка, существует ли файл
    if not os.path.exists(merged_report_path):
        # Создание пустого файла
        with open(merged_report_path, 'w'):
            pass

    if not os.path.exists(json_file):
        with open(json_file, 'w'):
            pass
