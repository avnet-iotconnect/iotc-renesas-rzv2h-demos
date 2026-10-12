# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Avnet
"""Minimal ctypes binding to the DRP-AI TVM runtime on the Renesas RZ/V2H.

The RZ/V AI SDK image ships ``libtvm_runtime.so`` (Renesas / EdgeCortix MERA
build with the ``mera_drp`` DRP-AI backend) but no Python ``tvm`` package and
no compiler. Renesas' own apps drive it from C++ through
``MeraDrpRuntimeWrapper``; this module is the same handful of PackedFunc
calls made over the stable TVM C API, so a DRP-AI TVM model directory
(``deploy.so`` / ``deploy.json`` / ``deploy.params``) can be run straight
from Python with nothing to cross-compile::

    from detectors.tvm_rt import DrpAiModel
    m = DrpAiModel("/home/weston/R01_object_detection/exe_v2h/yolov3_onnx")
    m.set_input(0, blob)          # float32 NCHW, matching m.input_shape
    m.run()
    outs = m.get_outputs()        # list of float32 numpy arrays

Mirrors ``apps/MeraDrpRuntimeWrapper.cpp`` from renesas-rz/rzv_drp-ai_tvm.
"""

import ctypes
import fcntl
import os
import struct

import numpy as np

LIB_CANDIDATES = ("libtvm_runtime.so", "/usr/lib/libtvm_runtime.so",
                  "/usr/lib64/libtvm_runtime.so")
DRPAI_DEV = "/dev/drpai0"

# TVMArgTypeCode (c_runtime_api.h)
_K_INT, _K_FLOAT, _K_HANDLE, _K_NULL = 0, 2, 3, 4
_K_DLTENSOR, _K_MODULE, _K_FUNC, _K_STR, _K_BYTES, _K_NDARRAY = 7, 9, 10, 11, 12, 13

# DLDeviceType
K_DL_CPU = 1
K_DL_DRPAI = 36

# linux/drpai.h: DRPAI_GET_DRPAI_AREA = _IOR(46, 11, drpai_data_t), where
# drpai_data_t is {u64 address; u64 size} on the RZ/V2H.
_DRPAI_DATA_SIZE = 16
_DRPAI_GET_DRPAI_AREA = (2 << 30) | (_DRPAI_DATA_SIZE << 16) | (46 << 8) | 11


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("device", _DLDevice),
                ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class _TVMValue(ctypes.Union):
    _fields_ = [("v_int64", ctypes.c_int64),
                ("v_float64", ctypes.c_double),
                ("v_handle", ctypes.c_void_p),
                ("v_str", ctypes.c_char_p)]


class _TVMByteArray(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("size", ctypes.c_size_t)]


class TVMError(RuntimeError):
    pass


_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    last = None
    for name in LIB_CANDIDATES:
        try:
            _lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as e:
            last = e
    if _lib is None:
        raise TVMError(f"DRP-AI TVM runtime (libtvm_runtime.so) not found: {last}")
    _lib.TVMGetLastError.restype = ctypes.c_char_p
    return _lib


def available():
    """True when both the TVM runtime and the DRP-AI device node exist."""
    if not os.path.exists(DRPAI_DEV):
        return False
    try:
        _load_lib()
        return True
    except TVMError:
        return False


def _check(rc):
    if rc != 0:
        raise TVMError(_lib.TVMGetLastError().decode("utf-8", "replace"))


def _call(func, *args):
    """Call a PackedFunc. Returns (python_value, type_code).

    Supported arg types: int, float, str, bytes-like via _TVMByteArray,
    _DLTensor (by pointer), and (handle, type_code) tuples."""
    n = len(args)
    values = (_TVMValue * max(n, 1))()
    codes = (ctypes.c_int * max(n, 1))()
    keep = []  # keep temporaries alive across the call
    for i, a in enumerate(args):
        if isinstance(a, bool) or isinstance(a, int):
            values[i].v_int64 = int(a)
            codes[i] = _K_INT
        elif isinstance(a, float):
            values[i].v_float64 = a
            codes[i] = _K_FLOAT
        elif isinstance(a, str):
            b = a.encode("utf-8")
            keep.append(b)
            values[i].v_str = b
            codes[i] = _K_STR
        elif isinstance(a, _TVMByteArray):
            values[i].v_handle = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)
            codes[i] = _K_BYTES
        elif isinstance(a, _DLTensor):
            values[i].v_handle = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)
            codes[i] = _K_DLTENSOR
        elif isinstance(a, tuple):
            values[i].v_handle = a[0]
            codes[i] = a[1]
        else:
            raise TypeError(f"unsupported PackedFunc arg {type(a)}")
    ret = _TVMValue()
    ret_code = ctypes.c_int()
    _check(_lib.TVMFuncCall(func, values, codes, ctypes.c_int(n),
                            ctypes.byref(ret), ctypes.byref(ret_code)))
    code = ret_code.value
    if code == _K_INT:
        return ret.v_int64, code
    if code == _K_FLOAT:
        return ret.v_float64, code
    if code == _K_NULL:
        return None, code
    if code == _K_STR:
        return ret.v_str.decode("utf-8", "replace"), code
    return ret.v_handle, code


def _global_func(name):
    h = ctypes.c_void_p()
    _check(_lib.TVMFuncGetGlobal(name.encode(), ctypes.byref(h)))
    return h if h.value else None


def drpai_area():
    """(start address, size) of the DRP-AI memory area (DRPAI_GET_DRPAI_AREA)."""
    fd = os.open(DRPAI_DEV, os.O_RDWR)
    try:
        buf = fcntl.ioctl(fd, _DRPAI_GET_DRPAI_AREA, bytes(_DRPAI_DATA_SIZE))
    finally:
        os.close(fd)
    return struct.unpack("QQ", buf)


def drpai_start_address():
    return drpai_area()[0]


# Every DRP-AI model loaded by this process needs its own slice of the DRP-AI
# memory area (Renesas' multi-model apps hard-code a per-model offset). The
# runtime doesn't report a model's footprint, so a slice is sized from
# deploy.so (weights dominate) with generous headroom. One model — the normal
# case — always sits at the head of the area, exactly like the Renesas apps.
_SLOT_ALIGN = 0x1000000  # 16 MB
_slots = []              # [(offset, size, owner_id)]


def _alloc_slot(owner, so_bytes):
    base, area = drpai_area()
    size = -(-(2 * so_bytes + 0x4000000) // _SLOT_ALIGN) * _SLOT_ALIGN
    offset = 0
    for o, sz, _ in sorted(_slots):
        if offset + size <= o:
            break
        offset = o + sz
    if _slots and offset + size > area:
        raise TVMError("DRP-AI memory area is full — close another model first")
    _slots.append((offset, size, id(owner)))
    return base + offset


def _free_slot(owner):
    _slots[:] = [s for s in _slots if s[2] != id(owner)]


# NumPy converts float16 in software (~10 ns per element on the Cortex-A55),
# which made reading a YOLOv8 model's outputs cost nearly as much as running
# it. A 65536-entry table indexed by the raw bit pattern is several times
# faster.
_F16_TO_F32 = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)


def f16_to_f32(a):
    """float16 array -> float32, via lookup table."""
    return _F16_TO_F32[a.view(np.uint16)]


_NP_DTYPES = {(2, 32): np.float32, (2, 16): np.float16, (2, 64): np.float64,
              (0, 64): np.int64, (0, 32): np.int32, (0, 8): np.int8,
              (1, 8): np.uint8}


class DrpAiModel:
    """One DRP-AI TVM model directory loaded onto the accelerator."""

    def __init__(self, model_dir, start_address=None):
        _load_lib()
        self.model_dir = model_dir
        for f in ("deploy.so", "deploy.json", "deploy.params"):
            if not os.path.exists(os.path.join(model_dir, f)):
                raise TVMError(f"{model_dir} is not a DRP-AI TVM model dir (missing {f})")

        with open(os.path.join(model_dir, "deploy.json")) as f:
            json_data = f.read()
        # Same rule as MeraDrpRuntimeWrapper: a graph with no DRP-AI subgraph
        # is a CPU-only TVM model.
        self.on_drpai = "drp" in json_data
        device_type = K_DL_DRPAI if self.on_drpai else K_DL_CPU

        self._f = {}
        self._mod = self._syslib = None
        mod = ctypes.c_void_p()
        so_path = os.path.join(model_dir, "deploy.so").encode()
        _check(_lib.TVMModLoadFromFile(so_path, b"so", ctypes.byref(mod)))
        self._syslib = mod

        create = (_global_func("tvm.graph_executor_debug.create")
                  or _global_func("tvm.graph_executor.create"))
        if create is None:
            raise TVMError("TVM runtime has no graph executor")
        handle, code = _call(create, json_data, (mod, _K_MODULE), device_type, 0)
        if code != _K_MODULE:
            raise TVMError(f"graph executor create returned type code {code}")
        self._mod = ctypes.c_void_p(handle)
        with open(os.path.join(model_dir, "deploy.params"), "rb") as f:
            params = f.read()
        arr = _TVMByteArray(params, len(params))
        _call(self._func("load_params"), arr)

        set_addr = self._func("set_start_address", required=False)
        if set_addr is not None and self.on_drpai:
            if start_address is None:
                start_address = _alloc_slot(
                    self, os.path.getsize(os.path.join(model_dir, "deploy.so")))
            _call(set_addr, int(start_address))

        self.num_outputs = int(_call(self._func("get_num_outputs"))[0])
        handle, code = _call(self._func("get_input"), 0)
        self.input_shape, self.input_dtype = self._describe(handle)
        if code == _K_NDARRAY:
            _lib.TVMArrayFree(ctypes.c_void_p(handle))
        self._in_keep = None

    def close(self):
        """Unload the model and give its DRP-AI memory slice back. The DRP-AI
        is a single shared accelerator: release it before another process
        (e.g. a calibrator subprocess) loads a model of its own."""
        if getattr(self, "_mod", None) is None:
            return
        for h in self._f.values():
            _lib.TVMFuncFree(h)
        self._f = {}
        _lib.TVMModFree(self._mod)
        _lib.TVMModFree(self._syslib)
        self._mod = self._syslib = None
        self._in_keep = None
        _free_slot(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _func(self, name, required=True):
        if name in self._f:
            return self._f[name]
        h = ctypes.c_void_p()
        _check(_lib.TVMModGetFunction(self._mod, name.encode(), 0, ctypes.byref(h)))
        if not h.value:
            if required:
                raise TVMError(f"graph executor has no function {name!r}")
            return None
        self._f[name] = h
        return h

    @staticmethod
    def _describe(handle):
        t = ctypes.cast(handle, ctypes.POINTER(_DLTensor)).contents
        shape = tuple(int(t.shape[i]) for i in range(t.ndim))
        dtype = _NP_DTYPES.get((t.dtype.code, t.dtype.bits))
        if dtype is None:
            raise TVMError(f"unsupported tensor dtype code={t.dtype.code} bits={t.dtype.bits}")
        return shape, dtype

    def set_input(self, index, array):
        """Bind a numpy array as graph input ``index`` (copied by the runtime)."""
        a = np.ascontiguousarray(array, dtype=self.input_dtype)
        if a.shape != self.input_shape:
            raise ValueError(f"input shape {a.shape} != model {self.input_shape}")
        shape = (ctypes.c_int64 * a.ndim)(*a.shape)
        t = _DLTensor()
        t.data = a.ctypes.data
        t.device = _DLDevice(K_DL_CPU, 0)
        t.ndim = a.ndim
        t.dtype = _DLDataType(2, a.dtype.itemsize * 8, 1)
        t.shape = shape
        t.strides = None
        t.byte_offset = 0
        self._in_keep = (a, shape, t)
        _call(self._func("set_input"), int(index), t)

    def run(self):
        _call(self._func("run"))

    def get_output(self, index, raw=False):
        """Output ``index`` copied out of the runtime: as float32, or with
        ``raw=True`` in the model's own dtype (DRP-AI heads are float16 —
        convert just the parts you need with ``f16_to_f32``)."""
        handle, code = _call(self._func("get_output"), int(index))
        try:
            t = ctypes.cast(handle, ctypes.POINTER(_DLTensor)).contents
            shape, dtype = self._describe(handle)
            n = int(np.prod(shape)) * np.dtype(dtype).itemsize
            buf = (ctypes.c_char * n).from_address(t.data + t.byte_offset)
            out = np.frombuffer(buf, dtype=dtype).reshape(shape)
            if raw:
                return out.copy()
            if dtype is np.float16:
                return f16_to_f32(out)
            return out.astype(np.float32)
        finally:
            if code == _K_NDARRAY:
                _lib.TVMArrayFree(ctypes.c_void_p(handle))

    def get_outputs(self, raw=False):
        return [self.get_output(i, raw=raw) for i in range(self.num_outputs)]

    def infer(self, array, raw=False):
        self.set_input(0, array)
        self.run()
        return self.get_outputs(raw=raw)
