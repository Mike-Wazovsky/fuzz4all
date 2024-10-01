import argparse
import glob
import os
import subprocess
import time

from KoverCoverage.kotlin_coverage import KotlinKover
from helper import (
    parse_xml,
    save_ic,
    move_ic,
    save_experiment_state,
    restore_experiment_state,
    check_ic,
    build_graphs, prepare, cleanup, delete, create_directory, merge_ic
)


def main():
    parser = argparse.ArgumentParser(description="Run the coverage function with specified parameters.")

    parser.add_argument("compiler", type=str, help="The target parameter for the coverage function")
    parser.add_argument("source", type=str, help="The folder parameter for the coverage function")
    parser.add_argument("output", type=str, help="output folder")
    parser.add_argument("size", type=int, default=-1, help="The number of .fuzz files")
    parser.add_argument("--scope", type=int, default=100,
                        help="The scope parameter for the coverage function (default: 100)")
    parser.add_argument("--restart", type=bool, default=False,
                        help="The restart parameter (default: False)")

    args = parser.parse_args()

    coverage(args.compiler, args.source, args.output, args.scope, args.size, args.restart)


def coverage(compiler: str, source: str, output: str, scope: int, size: int, restart: bool):
    global it
    target_dir = "KoverCoverage/run/kotlinc_tmp"
    json_file = os.path.join(output, "experiment.json")
    kotlinc_path = os.path.join(target_dir, "bin/kotlinc")
    target = KotlinKover(kotlinc_path)
    graphs = {
        "INSTRUCTION": [0],
        "BRANCH": [0],
        "LINE": [0],
        "METHOD": [0],
        "CLASS": [0]
    } if not restart else restore_experiment_state('graphs', json_file)
    start_iter = 0 if not restart else restore_experiment_state('iteration', json_file)
    x_axis = [0] if not restart else restore_experiment_state('x_axis', json_file)

    try:
        # добавить опцию JAVA_OPT
        os.environ["JAVA_OPTS"] = \
            ("-javaagent:KoverCoverage/Kover/sources/kover-jvm-agent-0.8.4-SNAPSHOT.jar=file:KoverCoverage/Kover/agent"
             ".args")

        # создать копию kotlin target
        prepare(compiler, target_dir, json_file, output)

        # Прогнать запуски kotlin target
        for it in range(start_iter, size):
            if (it % scope == 0 or it == size - 1) and it > x_axis[-1]:
                res = read_coverage(target_dir, output, it)
                for key, value in res.items():
                    graphs[key].append(int(value))
                x_axis.append(it)
            save_experiment_state(it, graphs, x_axis, json_file)

            file_path = os.path.join(source, f"{it}.fuzz")

            print(f"run {it}.fuzz")

            run_individual(target, file_path, it)
            save_ic(output)
            merge_ic(output)

        # Построить графики
        build_graphs(graphs, x_axis)

        # убрать опцию JAVA_OPT + удалить директорию kotlinc
        save_experiment_state(it, graphs, x_axis, json_file)
        cleanup(target_dir)
    except Exception as e:
        save_experiment_state(it, graphs, x_axis, json_file)
        cleanup(target_dir)

        raise e


def run_individual(target, file_path, num):
    target.run_individual(file_path)
    if not os.path.isfile("KoverCoverage/run/kover-kotlin-report.ic"):
        raise FileNotFoundError(f"Файл KoverCoverage/run/kover-kotlin-report.ic не найден.")

    # проверить, всё ли норм, если нет, то вернуть backup версию и запуститься заново


def read_coverage(target_dir: str, output: str, it: int) -> dict[str, int]:
    print(f"run xml")
    xml_path = "KoverCoverage/run/data.xml"

    command = [
        "java",
        "-jar",
        "KoverCoverage/Kover/sources/kover-cli-0.8.4-SNAPSHOT.jar",
        "report",
        # "KoverCoverage/run/kover-kotlin-report.ic",
        os.path.join(output, "merged-kover-kotlin-report.ic"),
        "--classfiles",
        os.path.join(target_dir, "lib/kotlin-compiler.jar"),
        "--src",
        "src/main/kotlin",
        "--xml",
        xml_path
    ]

    try:
        start_time = time.time()  # Record the start time

        subprocess.run(command, check=True, capture_output=True, text=True)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"Command executed in {elapsed_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print("Произошла ошибка при выполнении команды:")
        print(e.stderr)

    return parse_xml(xml_path)


if __name__ == "__main__":
    main()
