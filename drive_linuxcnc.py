import linuxcnc
import serial
from PySide6.QtCore import QObject
from PySide6.QtCore import Signal
import time
import json
import re


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class LinuxDriver(QObject):  # type: ignore
    OnSampleReceived = Signal(list)

    def __init__(self) -> None:
        super().__init__()

        self.s = linuxcnc.stat()  # type: ignore
        self.c = linuxcnc.command()  # type: ignore

    def ready(self) -> bool:

        self.s.poll()
        return (
            not self.s.estop
            and self.s.enabled
            and (self.s.homed.count(1) == self.s.joints)
            and (self.s.interp_state == linuxcnc.INTERP_IDLE)  # type: ignore
        )

    def cmd(self, cmd: str) -> None:
        self.c.mdi(cmd)
        print(f"Sent: {cmd}")
        self.c.wait_complete()  # wait until mode switch executed


d = LinuxDriver()


def main():
    # initialize serial port
    ser = serial.Serial()
    ser.port = "/dev/ttyUSB1"  # Arduino serial port
    ser.baudrate = 115200
    ser.timeout = 10  # specify timeout when using readline()
    ser.open()

    if ser.is_open == False:
        return

    d.c.mode(linuxcnc.MODE_MDI)  # type: ignore
    d.c.wait_complete()  # wait until mode switch executed

    rpm = 2000
    rpm_step = 10
    num_samples = 500
    wait = 0.5  # seconds

    d.cmd(f"S{rpm}")
    d.cmd("M3")

    time.sleep(4)

    data = {}
    data_filename = f"SpindlePlate_{rpm_step}_rpm_{num_samples}_samples"

    while rpm < 24000:
        d.cmd(f"S{rpm}")
        rpm += rpm_step

        # wait for the spindle to get the speed up
        time.sleep(wait)

        samples = []
        while len(samples) < num_samples:
            rcv = str(ser.readline())
            vals = re.findall(r"[-+]?(?:\d*\.*\d+)", rcv)
            if len(vals) < 3:
                print("samples issue")
                continue
            try:
                float_vals = [float(i) for i in vals][:3]
            except Exception as e:
                print(e, "float conversion or index error")
                continue
            samples.append(float_vals)

        data[str(rpm)] = samples

        if rpm % 500 == 0:
            with open(f"{data_filename}_{rpm}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            data = {}

    print("Exiting...")
    d.cmd("M5")


if __name__ == "__main__":
    main()
