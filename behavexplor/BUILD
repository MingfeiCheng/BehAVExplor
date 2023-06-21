load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "corpus",
    srcs = ["corpus.py"],
    deps = [
    "//BehAVExplor/behavexplor:element",
    "//BehAVExplor/behavexplor/models:coverage",
    ]
)

py_binary(
    name = "element",
    srcs = ["element.py"],
)

py_binary(
    name = "utils",
    srcs = ["utils.py"],
)