"""generate_qua_script text in one spelling across qm-qua versions.

qm-qua 1.4 prints string arguments with single quotes and a scaled play as
``play('op', 'el', amplitude_scale=x)``, where 1.2 printed ``play("op"*amp(x), "el")``.
Tests that read the script go through qua_text so their patterns hold on both.
"""

import re

_SCALED_PLAY = re.compile(r'play\(("[^"]+"), ("[^"]+"), amplitude_scale=([^,()]+)(\)|, )')


def qua_text(prog, config=None) -> str:
    from qm import generate_qua_script

    text = generate_qua_script(prog, config) if config is not None else generate_qua_script(prog)
    text = text.replace("'", '"')
    return _SCALED_PLAY.sub(r"play(\1*amp(\3), \2\4", text)
