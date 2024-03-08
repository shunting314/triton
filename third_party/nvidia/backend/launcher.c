#include "cuda.h"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{
   if (code != CUDA_SUCCESS)
   {
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {0};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {
  // Open the shared library
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (!handle) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");
    return NULL;
  }
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so");
    return NULL;
  }
  return cuLaunchKernelExHandle;
}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function, void** params) {
  if (gridX*gridY*gridZ > 0) {
    if (num_ctas == 1) {
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    } else {
      CUlaunchAttribute launchAttr[2];
      launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = clusterDimX;
      launchAttr[0].value.clusterDim.y = clusterDimY;
      launchAttr[0].value.clusterDim.z = clusterDimZ;
      launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      CUlaunchConfig config;
      config.gridDimX = gridX * clusterDimX;
      config.gridDimY = gridY * clusterDimY;
      config.gridDimZ = gridZ * clusterDimZ;
      config.blockDimX = 32 * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
      if (cuLaunchKernelExHandle == NULL) {
        cuLaunchKernelExHandle = getLaunchKernelExHandle();
      }
      CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
    }
  }
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  PyObject *args_obj = NULL;

  if(!PyArg_ParseTuple(args, "iiiiiiiiiKKOOOO!", &gridX, &gridY, &gridZ, &num_warps, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, &PyTuple_Type, &args_obj)) {
    return NULL;
  }

  Py_ssize_t nargs = PyTuple_Size(args_obj);

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {
    return NULL;
  }

  uint64_t* arg_vals = malloc(nargs * sizeof(uint64_t));
  void** arg_ptrs = malloc(nargs * sizeof(void*));

  for (Py_ssize_t i = 0; i < nargs; ++i) {
    PyObject *item = PyTuple_GetItem(args_obj, i);
    arg_vals[i] = PyLong_AsUnsignedLongLong(item);
    arg_ptrs[i] = &arg_vals[i];
  }

  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function, arg_ptrs);

  free(arg_vals);
  free(arg_ptrs);

  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {
    return NULL;
  }

  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {
    return NULL;
  }

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
