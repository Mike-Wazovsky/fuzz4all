import re
import subprocess

from Fuzz4All.target.target import FResult, Target
from Fuzz4All.util.util import comment_remover


class KotlinTarget(Target):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs["template"] == "fuzzing_with_config_file":
            config_dict = kwargs["config_dict"]
            self.prompt_used = self._create_prompt_from_config(config_dict)
            self.config_dict = config_dict
        else:
            raise NotImplementedError

    def write_back_file(self, code, write_back_name=""):
        if write_back_name != "":
            try:
                with open(write_back_name, "w", encoding="utf-8") as f:
                    f.write(code)
            except:
                pass
        return "/tmp/temp{}.kt".format(self.CURRENT_TIME)

    def wrap_prompt(self, prompt: str) -> str:
        return f"/* {prompt} */\n{self.prompt_used['separator']}\n{self.prompt_used['begin']}"

    def wrap_in_comment(self, prompt: str) -> str:
        return f"/* {prompt} */"

    def filter(self, code: str) -> bool:
        clean_code = code.replace(self.prompt_used["begin"], "").strip()
        if self.prompt_used["target_api"] not in clean_code:
            return False
        return True

    def clean(self, code: str) -> str:
        code = self.extract_java_code(code)
        code = comment_remover(code)
        return code

    def clean_code(self, code: str) -> str:
        code = self.extract_java_code(code)
        code = comment_remover(code)
        code = "\n".join(
            [
                line
                for line in code.split("\n")
                if line.strip() != "" and line.strip() != self.prompt_used["begin"]
            ]
        )
        return code

    def extract_java_code(self, code: str):
        # pattern = re.compile(r'```\n(.*?)\n```', re.DOTALL)
        pattern = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
        match = pattern.search(code)

        if match:
            return match.group(1)
        else:
            return code

    # If there exists a public class, ensure file name matches
    def determine_file_name(self, code):
        # return standard write back file name
        return "/tmp/temp{}.kt".format(self.CURRENT_TIME)

    def validate_individual(self, filename) -> (FResult, str):
        write_back_name = ""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
                write_back_name = self.determine_file_name(code)
                self.write_back_file(code, write_back_name=write_back_name)
        except:
            pass

        try:
            exit_code = subprocess.run(
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
            return FResult.TIMED_OUT, "java"
        if exit_code.returncode == 1:
            return FResult.FAILURE, exit_code.stderr
        elif exit_code.returncode == 0:
            return FResult.SAFE, "its safe"
        else:
            return FResult.ERROR, exit_code.stderr
