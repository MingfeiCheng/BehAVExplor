load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "coverage",
    srcs = ["coverage.py"],
    deps= [
        "//BehAVExplor/behavexplor/models:feature",
    ]
)

py_binary(
    name = "feature",
    srcs = ["feature.py"],
)
