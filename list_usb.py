# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""List USB devices (VID:PID, names) from sysfs — the image has no lsusb.
The xArm shows up as 0483:5750 (Hiwonder xArm)."""
import glob
import os


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def main():
    print("USB Devices:")
    for d in sorted(glob.glob("/sys/bus/usb/devices/*")):
        vid = _read(os.path.join(d, "idVendor"))
        if not vid:
            continue
        print(f"{os.path.basename(d):8} VID: 0x{vid}, PID: 0x{_read(os.path.join(d, 'idProduct'))}"
              f"  {_read(os.path.join(d, 'manufacturer'))} {_read(os.path.join(d, 'product'))}")


if __name__ == "__main__":
    main()
