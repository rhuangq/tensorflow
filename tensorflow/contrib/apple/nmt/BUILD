# Description:
# Example neural translation models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils_qnn",
        ":seq2seq_attention",
    ],
)

py_library(
    name = "data_utils_qnn",
    srcs = [
        "data_utils_qnn.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "seq2seq_attention",
    srcs = [
        "seq2seq_attention.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
    ":data_utils_qnn",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "seq2seq_train",
    srcs = [
        "seq2seq_train.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils_qnn",
        ":seq2seq_attention",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "seq2seq_create_eval_graph",
    srcs = [
        "seq2seq_create_eval_graph.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils_qnn",
        ":seq2seq_attention",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "seq2seq_inference",
    srcs = [
        "seq2seq_inference.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils_qnn",
        ":seq2seq_attention",
        "//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
