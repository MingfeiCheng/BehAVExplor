load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "//BehAVExplor/behavexplor:corpus",
        "//BehAVExplor/behavexplor:utils",
        "//BehAVExplor/common:scenario",
        "//BehAVExplor/common:runner",
        "//BehAVExplor/common:simulator",
    ],
)