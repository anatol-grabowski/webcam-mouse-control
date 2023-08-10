
def list_webcams():
    from collections import defaultdict
    import re
    import subprocess

    # Command as a list of strings

    completed_process = subprocess.run(
        'v4l2-ctl --list-devices 2>/dev/null',
        shell=True, stdout=subprocess.PIPE, text=True
    )

    stdout_output = completed_process.stdout
    # print("Stdout Output:")
    # print(stdout_output)

    device_info = defaultdict(list)
    current_device = ""

    for line in stdout_output.splitlines():
        line = line.strip()
        if line:
            if re.match(r"^\w+.*:", line):
                current_device = line
            else:
                device_info[current_device].append(line)

    parsed_dict = dict(device_info)

    # print(parsed_dict)
    return parsed_dict
