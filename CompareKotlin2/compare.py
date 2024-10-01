import os

import click
from rich.traceback import install

from CompareKotlin2.comp.KotlinComp import KotlinComp

install()


def write_to_file(fo, file_name):
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(fo)
    except:
        pass


def compare(
    target: KotlinComp,
    fuzz_folder: str,
):

    count = 0
    val_time = 0

    for filename in os.listdir(fuzz_folder):
        if filename.endswith('.fuzz'):
            file_path = os.path.join(fuzz_folder, filename)
            try:
                c_result, message = target.validate_individual(file_path)
                target.parse_validation_message(c_result, message, file_path)
                # print(f"File: {filename}, Result: {c_result}, Message: {message}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")


# evaluate against the oracle to discover any potential bugs
# used after the generation
def evaluate_all(target: KotlinComp):
    target.validate_all()


@click.group()
@click.pass_context
def cli(ctx):
    """Run the main using a configuration file."""
    ctx.ensure_object(dict)


@cli.command("main_with_config")
@click.pass_context
@click.option(
    "folder",
    "--folder",
    type=str,
    default="Results/test",
    help="folder to store results",
)
@click.option(
    "target_1",
    "--target_1",
    type=str,
    default="",
    help="specific target to run",
)
@click.option(
    "target_2",
    "--target_2",
    type=str,
    default="",
    help="specific target to run",
)
def main_with_config(ctx, folder, target_1, target_2):
    target = KotlinComp(folder=folder, target_name_1=target_1, target_name_2=target_2)

    compare(target=target,
            fuzz_folder=folder,)

if __name__ == "__main__":
    cli()