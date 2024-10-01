import glob
import subprocess
import time
from enum import Enum

from rich.progress import track

from Fuzz4All.target.target import FResult
from Fuzz4All.util.Logger import LEVEL, Logger


class CompResult(Enum):
    EQUAL = 1  # validation returns okay
    UNEQUAL = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    TIMED_OUT = 10  # timed out, can be okay in certain targets


class KotlinComp():
    def __init__(self, folder="/", **kwargs):
        self.folder = folder
        self.CURRENT_TIME = time.time()
        # loggers
        self.g_logger = Logger(self.folder, "log_generation.txt", level=LEVEL.VERBOSE)
        self.v_logger = Logger(self.folder, "log_compare.txt", level=LEVEL.VERBOSE)
        # main logger for system messages
        self.m_logger = Logger(self.folder, "log.txt")
        if "target_name_1" in kwargs:
            self.target_name_1 = kwargs["target_name_1"]
        if "target_name_2" in kwargs:
            self.target_name_2 = kwargs["target_name_2"]

    def write_back_file(self, code, write_back_name=""):
        if write_back_name != "":
            try:
                with open(write_back_name, "w", encoding="utf-8") as f:
                    f.write(code)
            except:
                pass
        return "/tmp/temp{}.kt".format(self.CURRENT_TIME)

    # If there exists a public class, ensure file name matches
    def determine_file_name(self, code):
        # return standard write back file name
        return "/tmp/temp{}.kt".format(self.CURRENT_TIME)

    def parse_validation_message(self, c_result, message, file_name):
        # TODO: rewrite to include only status in TRACE but full message in VERBOSE
        self.v_logger.logo("Comparing {} ...".format(file_name), LEVEL.TRACE)
        if c_result == CompResult.EQUAL:
            self.v_logger.logo("{} is equal".format(file_name), LEVEL.VERBOSE)
        elif c_result == CompResult.UNEQUAL:
            self.v_logger.logo(
                "{} validations return different codes: {}".format(
                    file_name, message, LEVEL.VERBOSE
                )
            )
        elif c_result == CompResult.TIMED_OUT:
            self.v_logger.logo("{} timed out".format(file_name), LEVEL.VERBOSE)

    def validate_individual(self, filename) -> (CompResult, str):
        write_back_name = ""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
                write_back_name = self.determine_file_name(code)
                self.write_back_file(code, write_back_name=write_back_name)
        except:
            pass

        try:
            exit_code_1 = subprocess.run(
                f"{self.target_name_1} {write_back_name} -d out",
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=20,
                text=True,
            )

            exit_code_2 = subprocess.run(
                f"{self.target_name_2} {write_back_name} -d out",
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=20,
                text=True,
            )
        except subprocess.TimeoutExpired as te:
            pname = f"'temp{self.CURRENT_TIME}'"
            subprocess.run(
                ["ps -ef | grep " + pname + " | grep -v grep | awk '{print $2}'"],
                shell=True,
            )
            subprocess.run(
                [
                    "ps -ef | grep "
                    + pname
                    + " | grep -v grep | awk '{print $2}' | xargs -r kill -9"
                ],
                shell=True,
            )  # kill all tests thank you
            return CompResult.TIMED_OUT, "java"
        f_res_1, message_1 = self.code_to_result(exit_code_1)
        f_res_2, message_2 = self.code_to_result(exit_code_2)
        if f_res_1 == f_res_2:
            return CompResult.EQUAL, f'result: {message_1}'
        elif f_res_1 != f_res_2:
            return CompResult.UNEQUAL, f'result of first: {message_1}, second: {message_2}'



    def code_to_result(self, code) -> (FResult, str):
        if code.returncode == 1:
            return FResult.FAILURE, "Failure"
        elif code.returncode == 0:
            return FResult.SAFE, "Safe"
        else:
            return FResult.ERROR, "Error"

    def validate_all(self):
        for fuzz_output in track(
            glob.glob(self.folder + "/*.fuzz"),
            description="Validating",
        ):
            f_result, message = self.validate_individual(fuzz_output)
            self.parse_validation_message(f_result, message, fuzz_output)
