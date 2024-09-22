"""
Scripts for modifying training data class ids, to not override
the preexisting model's classes.
"""

import os




PATH = r"C:\development\GIT\robotski-vid\dataset\valid\labels"

MAPPING = {
    0: 80, # Box
    1: 81, # hanging-rod
    2: 0,   # Person
    3: 82, # small-floor-item
}

for path, dirs, files in os.walk(PATH):
    for file in files:
        file = os.path.join(path, file)

        if not file.endswith('.txt'):
            continue

        with open(file, "r") as f:
            data = f.read()
            if not data:
                continue

            rows = data.split('\n')
            for i, row in enumerate(rows):
                data = row.split(' ')
                cls = int(data[0])
                data[0] = str(MAPPING.get(cls, cls))
                rows[i] = ' '.join(data)

            data = '\n'.join(rows)

        with open(file, "w") as f:
            f.write(data)

