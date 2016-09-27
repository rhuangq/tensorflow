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
        ":data_utils",
        ":data_utils_qnn",
        ":seq2seq_model",
        ":seq2seq_attention",
    ],
)

py_library(
    name = "data_utils",
    srcs = [
        "data_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
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
    name = "seq2seq_model",
    srcs = [
        "seq2seq_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        "//tensorflow:tensorflow_py",
    ],
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


py_binary(
    name = "translate",
    srcs = [
        "translate.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        ":seq2seq_model",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "translate_test",
    size = "medium",
    srcs = [
        "translate.py",
    ],
    args = [
        "--self_test=True",
    ],
    main = "translate.py",
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        ":seq2seq_model",
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
