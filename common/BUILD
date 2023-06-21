load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "frame",
    srcs = ["frame.py"],
    deps= [
        "//BehAVExplor/common:utils",
    ]
)

py_binary(
    name = "runner",
    srcs = ["runner.py"],
)

py_binary(
    name = "scenario",
    srcs = ["scenario.py"],
    deps= [
        "//BehAVExplor/common:utils",
    ]
)

py_binary(
    name = "simulator",
    srcs = ["simulator.py"],
    deps= [
        "//BehAVExplor/common:utils",
        "//BehAVExplor/common:frame",
    ]
)


py_binary(
    name = "utils",
    srcs = ["utils.py"],
)



