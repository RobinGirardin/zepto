#!/usr/bin/env Rscript
# Wrapper around the shared sample-size script.
file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
this_dir <- dirname(normalizePath(file_arg[[1]]))
source(normalizePath(file.path(this_dir, "..", "..", "validity_common", "r", "power.R")))
