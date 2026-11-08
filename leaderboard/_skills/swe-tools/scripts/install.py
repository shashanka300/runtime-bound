"""Copy the sibling helper scripts to /tmp/tools (run once, no arguments)."""
import os
import shutil

here = os.path.dirname(os.path.abspath(__file__))
dest = "/tmp/tools"
os.makedirs(dest, exist_ok=True)
names = []
for name in sorted(os.listdir(here)):
    if name.endswith(".py") and name != "install.py":
        shutil.copy(os.path.join(here, name), os.path.join(dest, name))
        names.append(name)
print("installed: " + ", ".join(names) + " in /tmp/tools")
print('next: python /tmp/tools/locate.py "<identifiers and key words from the issue>"')
