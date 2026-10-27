"""Inyección de ratón con SendInput (API oficial de Windows; no requiere driver)."""
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
MOVE, ABS = 0x0001, 0x8000
BUTTONS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _send(flags, dx=0, dy=0):
    i = INPUT(type=0, mi=MOUSEINPUT(dx, dy, 0, flags, 0, 0))
    user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(INPUT))


def move(x, y):
    w, h = screen_size()
    _send(MOVE | ABS, int(x * 65535 / max(w - 1, 1)), int(y * 65535 / max(h - 1, 1)))


def down(button="left"):
    _send(BUTTONS[button][0])


def up(button="left"):
    _send(BUTTONS[button][1])


def click(button="left"):
    down(button); up(button)


def double():
    click(); click()


def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def hotkey_down():
    """F9 o Ctrl+Alt+P: parada de emergencia global (pausa/reanuda la inyección)."""
    return key_down(0x78) or (key_down(0x11) and key_down(0x12) and key_down(0x50))
