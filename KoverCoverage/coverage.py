import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import xmltodict

from KoverCoverage.kotlin_coverage import KotlinKover


def main():
    parser = argparse.ArgumentParser(description="Run the coverage function with specified parameters.")

    parser.add_argument("compiler", type=str, help="The target parameter for the coverage function")
    parser.add_argument("source", type=str, help="The folder parameter for the coverage function")
    parser.add_argument("size", type=int, default=-1, help="The number of .fuzz files")
    parser.add_argument("--scope", type=int, default=100,
                        help="The scope parameter for the coverage function (default: 100)")
    parser.add_argument("--restart", type=bool, default=False,
                        help="The restart parameter (default: False)")

    args = parser.parse_args()

    coverage(args.compiler, args.source, args.scope, args.size, args.restart)


def coverage(compiler: str, source: str, scope: int, size: int, restart: bool):
    global iter
    target_dir = "KoverCoverage/Kover/kotlinc_tmp"
    json_dir = "KoverCoverage/Kover/experiment.json"
    kotlinc_path = os.path.join(target_dir, "bin/kotlinc")
    target = KotlinKover(kotlinc_path)
    graphs = {
        "INSTRUCTION": [0],
        "BRANCH": [0],
        "LINE": [0],
        "METHOD": [0],
        "CLASS": [0]
    } if not restart else restore_experiment_state('graphs', json_dir)
    start_iter = 0 if not restart else restore_experiment_state('iteration', json_dir)
    x_axis = [0] if not restart else restore_experiment_state('x_axis', json_dir)

    try:
        #добавить опцию JAVA_OPT
        os.environ["JAVA_OPTS"] = "-javaagent:KoverCoverage/Kover/kover-jvm-agent-0.8.3.jar=file:KoverCoverage/Kover/agent.args"

        #создать копию kotlin target
        if not restart:
            delete("KoverCoverage/Kover/kover-kotlin-report.ic")
            delete("KoverCoverage/Kover/data.xml")
            delete(target_dir)

            if not os.path.isdir(compiler):
                print(f"given '{compiler}' doesn't exist")
            else:
                try:
                    shutil.copytree(compiler, target_dir)
                except Exception as e:
                    print(f"Error occurred during copying of target: {e}")

        #Прогнать запуски kotlin target
        for iter in range(start_iter, size):
            if iter % scope == 0 and iter > x_axis[-1]:
                res = read_coverage(target_dir)
                for key, value in res.items():
                    graphs[key].append(int(value))
                x_axis.append(iter)

            file_path = os.path.join(source, f"{iter}.fuzz")
            print(f"run {iter}.fuzz")
            # if (iter == 6):
            #     raise Exception(f"RANDOM ERROR!!!!!!!")
            target.run_individual(file_path)

        #Снять результат
        # res = read_coverage(target_dir)
        # for key, value in res.items():
        #     graphs[key].append(int(value))
        #     x_axis.append(size)

        #Построить графики
        for key, values in graphs.items():
            fig, ax = plt.subplots()
            ax.plot(x_axis, values, label=key)

            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'{key} Coverage')
            ax.legend()

            plt.show()


        # убрать опцию JAVA_OPT + удалить директорию kotlinc
        save_experiment_state(iter, graphs, x_axis, json_dir)
        os.environ["JAVA_OPTS"] = ""
        delete(target_dir)
        delete("KoverCoverage/Kover/kover-kotlin-report.ic")
        delete("KoverCoverage/Kover/data.xml")
    except Exception as e:
        # TODO(save results)
        save_experiment_state(iter, graphs, x_axis, json_dir)

        os.environ["JAVA_OPTS"] = ""
        raise e



def read_coverage(target_dir: str) -> dict[str, int]:
    print(f"run xml")
    xml_path = "KoverCoverage/Kover/data.xml"
    command = [
        "java",
        "-jar",
        "KoverCoverage/Kover/kover-cli-0.8.4-SNAPSHOT.jar",
        "report",
        "KoverCoverage/Kover/kover-kotlin-report.ic",
        "--classfiles",
        os.path.join(target_dir, "lib/kotlin-compiler.jar"),
        "--src",
        "src/main/kotlin",
        "--xml",
        xml_path
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Произошла ошибка при выполнении команды:")
        print(e.stderr)

    return parse_xml(xml_path)

def parse_xml(xml_path) -> dict[str, int]:
    res = {}
    with open(xml_path) as file:
        data = xmltodict.parse(file.read())
        res = {x['@type']: x['@covered'] for x in data['report']['counter']}
    print(res)
    return res


def delete(path: str):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Произошла ошибка при удалении директории: {e}")


def save_experiment_state(iteration, graphs, x_axis, file_name='experiment.json'):
    state = {
        'iteration': iteration,
        'graphs': dict(graphs),
        'x_axis': x_axis
    }
    with open(file_name, 'w') as f:
        json.dump(state, f, indent=4)


def load_experiment_state(file_name='experiment.json'):
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


if __name__ == "__main__":
    main()