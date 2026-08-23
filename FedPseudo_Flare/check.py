import inspect
import nvflare
from nvflare.job_config.script_runner import ScriptRunner

print("NVFlare version:", nvflare.__version__)
print("ScriptRunner signature:")
print(inspect.signature(ScriptRunner))
