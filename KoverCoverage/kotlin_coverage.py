import glob
import subprocess
import time

from rich.progress import track

from Fuzz4All.target.target import FResult
from Fuzz4All.util.Logger import LEVEL, Logger


class KotlinKover:
    def __init__(self, target_name):
        self.CURRENT_TIME = time.time()
        self.target_name = target_name

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


    def run_individual(self, filename):
        write_back_name = ""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
                write_back_name = self.determine_file_name(code)
                self.write_back_file(code, write_back_name=write_back_name)
        except:
            pass

        try:
            subprocess.run(
                f"{self.target_name} {write_back_name} -d out",
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
            return



    def code_to_result(self, code) -> (FResult, str):
        if code.returncode == 1:
            return FResult.FAILURE, "Failure"
        elif code.returncode == 0:
            return FResult.SAFE, "Safe"
        else:
            return FResult.ERROR, "Error"
